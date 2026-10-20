"""Frame-by-frame absolute localization: matcher + estimators + a small state machine.

    LOST ──(global fixes)──▶ CONVERGING ──(one hypothesis wins)──▶ TRACKING
     ▲                          │                                    │
     └──────(all rejected)──────┴───────(no match for too long)───────┘

A wrong-but-confident position is worse than admitting the aircraft is lost.
Consecutive frames overlap almost completely, so a look-alike place that fools
one frame fools the next few as well. Instead of trusting the best global fix,
the localizer keeps several candidate locations, follows each one with local
searches as the aircraft flies over new ground, and only commits when one
candidate has clearly out-scored the others over a minimum distance.

A confirmed track is a sliding window of recent frames (``nav.window``) solved
jointly from odometry, heading and map matches, so a look-alike match is
outvoted by the rest of the window. When a track is lost, its window keeps
dead reckoning: the position is still reported (as coasting) with an
uncertainty that grows honestly, and the search for a new track is confined
to where that uncertainty allows. For recorded footage, ``smooth`` solves the
whole flight at once, bridging gaps from both sides.

Over a map with terrain, barometric heights above the takeoff point become
heights above the ground: at the current position for odometry, and wherever
the aircraft might be for matching. That needs the takeoff's elevation: from
the takeoff point when it is known, else from the first confirmed track, whose
matches measure the height above the ground they saw.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from itertools import pairwise

import numpy as np

from pharos.geomap import GeoMap
from pharos.geometry import Camera, rot2, wrap_angle
from pharos.nav.filter import FilterConfig, PoseFilter
from pharos.nav.window import PoseWindow, WindowConfig
from pharos.telemetry import ON_GROUND_M
from pharos.types import Estimate, Fix, Frame, HeightField, Status, Telemetry
from pharos.vision.matcher import (
    CorrelationMatcher,
    Matcher,
    MatcherConfig,
    SearchRegion,
    relief_at,
    view_height,
)
from pharos.vision.odometry import Motion, OdometryConfig, Relief, VisualOdometry
from pharos.vision.rectify import footprint


@dataclass(frozen=True)
class LocalizerConfig:
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    odometry: OdometryConfig | None = field(default_factory=OdometryConfig)  # None: off
    # Frames from lower or with the camera tilted further from straight down are skipped:
    # on the ground, taking off, or looking at the horizon.
    min_altitude_m: float = 15.0
    max_tilt_deg: float = 60.0
    # Beyond this tilt only odometry runs. On a real fixed-wing, half the confident matches
    # above 20 degrees of bank were 20+ m off, all of them while rolling fast, against 9%
    # near nadir. A camera held steadily oblique (a gimbal) still matches up to the limit.
    max_match_tilt_deg: float = 30.0
    max_match_tilt_rolling_deg: float = 20.0  # the limit while the tilt is changing...
    rolling_deg_s: float = 5.0  # ...faster than this
    # Acquisition
    prior_center: tuple[float, float] | None = None  # e.g. last GNSS fix before denial
    prior_radius_m: float = float("inf")
    compass_prior_deg: float = 45.0  # heading search half-width around the compass when lost
    init_min_score: float = 0.25
    init_min_distinctiveness: float = 0.1  # a winner must once have stood out map-wide
    max_hypotheses: int = 4
    lost_search_every: int = 2  # frames between global searches while lost
    reacquire_every: int = 6  # frames between global searches while converging
    # A lost track keeps dead reckoning. Acquisition then searches its 3-sigma region, and
    # every wide_search_every-th search the whole region (the lost track may have been wrong
    # already). For reacquire_window_s after the loss, while the region is small, a candidate
    # inside it is confirmed with fewer matches: continuity is evidence too.
    reacquire_window_s: float = 60.0
    reacquire_radius_m: float = 1000.0
    reacquire_confirm_frames: int = 3  # matches including the acquisition (vs confirm_frames)
    wide_search_every: int = 3
    max_dead_reckoning_m: float = 2000.0  # beyond this 3-sigma radius, dead reckoning stops
    max_speed: float = 30.0  # m/s; faster odometry is a wrong fit
    # Confirmation
    confirm_frames: int = 4
    confirm_distance_m: float = 120.0  # ground covered before a candidate can win
    confirm_margin: float = 0.06  # mean-score lead over the runner-up
    # Tracking
    track_min_score: float = 0.18
    # A match must stand out from its window's noise. With the frame-sized template, 2 keeps
    # fewer chance peaks than 1.5 did with the disk and still finds more true ones.
    track_min_excess: float = 2.0
    max_coast_converging_s: float = 5.0  # a candidate dies after this long without a match
    # A confirmed track keeps searching around its prediction (reported as coasting) while
    # the window that needs stays within max_search_m, for up to max_coast_s. Longer coasts
    # with wider searches were tried on real flights: they accept more look-alikes.
    max_coast_s: float = 30.0
    # An untrusted track that rejects this many matches within contradiction_window_s, all
    # pointing to one place, gives up its belief: it dead-reckons, and the place has to win
    # confirmation like any candidate. Counted as one, a run of matches cannot drag a wrong
    # window over, so without this the track would wait out max_coast_s. (Requiring full
    # confirmation here instead, as for a cold start, cost a third of flight 2's coverage.)
    contradiction_matches: int = 3
    contradiction_window_s: float = 10.0
    trust_sigma_m: float = 10.0  # beyond this uncertainty a track is reported as coasting
    # A position is trusted only with this many independent inlier matches in the window
    # (matches closer than reconfirm_distance_m of flight count once).
    trust_support: int = 2
    # Trusting a track again after coasting takes consecutive matches spread over this much
    # flown ground: overlapping frames see the same look-alike, so they are not independent.
    # Untrusted, a match must stand out more and counts less; a track that stays
    # untrusted for max_coast_s is dropped however many weak matches keep it company.
    reconfirm_frames: int = 2
    reconfirm_distance_m: float = 40.0
    untrusted_extra_excess: float = 1.0
    untrusted_meas_inflation: float = 4.0
    meas_sigma_m: float = 3.0
    meas_heading_sigma_deg: float = 2.0
    search_sigmas: float = 3.0
    min_search_m: float = 15.0
    max_search_m: float = 150.0
    min_heading_span_deg: float = 6.0  # at least three fine heading steps
    max_heading_span_deg: float = 30.0
    smoothing: float = 0.2  # EMA weight for altitude-scale and compass-offset estimates


def usable_view(tel: Telemetry, cfg: LocalizerConfig) -> bool:
    """Whether a frame shows the ground: high enough, and not looking at the horizon."""
    return tel.altitude >= cfg.min_altitude_m and tel.tilt <= np.radians(cfg.max_tilt_deg)


def matchable_view(tel: Telemetry, cfg: LocalizerConfig) -> bool:
    """Whether a frame is worth matching against the map, not only for odometry."""
    return usable_view(tel, cfg) and tel.tilt <= np.radians(cfg.max_match_tilt_deg)


@dataclass
class _Hypothesis:
    filter: PoseFilter
    scores: list[float] = field(default_factory=list)
    last_hit_t: float = 0.0
    last_hit_xy: np.ndarray | None = None
    travelled: float = 0.0  # distance between matched positions, not dead reckoning
    distinct: float = 0.0  # best map-wide distinctiveness seen when (re)acquired here
    trusted: bool = True  # False after coasting, until re-confirmed by consecutive matches
    near_track: bool = False  # found where a lost track's dead reckoning puts the aircraft
    streak: int = 0  # consecutive matches
    flown: float = 0.0  # dead-reckoned distance, metres
    streak_from: float = 0.0  # ``flown`` at the first match of the streak
    untrusted_since: float = 0.0  # time trust was lost
    frames: list[int] = field(default_factory=list)  # log indices since acquisition
    fixes: dict[int, Fix] = field(default_factory=dict)  # its matches, by log index

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

    @property
    def xy(self) -> np.ndarray:
        return self.filter.x[:2].copy()


class Localizer:
    def __init__(
        self,
        geomap: GeoMap,
        camera: Camera,
        cfg: LocalizerConfig | None = None,
        matcher: Matcher | None = None,
        takeoff: tuple[float, float] | None = None,
    ):
        """``takeoff`` is where the aircraft took off (world coordinates), if known."""
        self.cfg = cfg or LocalizerConfig()
        self.map = geomap
        self.camera = camera
        self.matcher = matcher or CorrelationMatcher(geomap, camera, self.cfg.matcher)
        self.terrain = geomap.terrain
        self.takeoff_elevation: float | None = None  # of the ground at takeoff, m
        if takeoff is not None:
            self.set_takeoff(takeoff)
        self._height: HeightField | None = None  # this frame's height above ground anywhere
        odometry = self.cfg.odometry
        self.odometry = VisualOdometry(camera, odometry) if odometry is not None else None
        self._motion: Motion | None = None
        self.status = Status.LOST
        self.hypotheses: list[_Hypothesis] = []
        self.scale = 1.0  # running estimate of true / reported altitude
        self.compass_offset = 0.0  # running estimate of true heading - compass
        self._frames_in_state = 0
        self._last_t: float | None = None
        self._last_yaw: float | None = None
        self.window: PoseWindow | None = None
        self.dead_reckoning: PoseWindow | None = None  # a lost track's window, matches stopped
        self._lost_at = 0.0
        self._airborne = False  # has left the ground (risen above takeoff)
        self._start: tuple | None = None  # a known position to begin from (start_at)
        self._started: tuple | None = None  # (log index, pose, covariance) it began at
        self._rejected: list[tuple[float, np.ndarray]] = []  # (time, offset from the track)
        self._searches = 0  # acquisition searches since the loss
        self.odometry_scale = 1.0  # learned by the window, carried to the next track
        self._window_start = 0.0
        # Every frame, for the whole-flight solve: telemetry, odometry, the match used, estimate.
        self.log: list[dict] = []

    def set_takeoff(self, xy: tuple[float, float]) -> None:
        """Where the aircraft took off (world coordinates): heights then follow the terrain."""
        if self.terrain is not None and np.isfinite(elevation := self.terrain.at(xy)):
            self.takeoff_elevation = float(elevation)

    def process(self, frame: Frame) -> Estimate:
        start = time.perf_counter()
        self._height = self._height_field(frame.telemetry)
        belief = self._belief(frame.t) if self._height is not None else None
        frame = self._above_ground(frame, belief)
        fix = None
        usable = usable_view(frame.telemetry, self.cfg)
        matchable = matchable_view(frame.telemetry, self.cfg) and not self._rolling(frame)
        self._motion = None
        if self.odometry is not None:
            if usable:
                relief = self._relief(frame, belief)
                motion = self.odometry.step(frame, self.scale, relief)
                self._motion = self._plausible(motion, frame.t)
            else:
                self.odometry.reset()
        if not matchable:
            for h in self.hypotheses:
                h.streak = 0  # a frame not matched breaks a run of matches
        self.log.append({"t": frame.t, "tel": frame.telemetry, "motion": self._motion, "fix": None})
        for h in self.hypotheses:
            h.frames.append(len(self.log) - 1)
        if self._start is not None:  # a known position: a track begins at this frame
            self._begin(frame)
        else:
            fix = self._step(frame, matchable)
        self._last_t = frame.t
        self._last_yaw = frame.telemetry.yaw
        return self._estimate(frame.t, fix, (time.perf_counter() - start) * 1e3)

    def start_at(
        self,
        xy: tuple[float, float],
        sigma_m: float = 5.0,
        heading: float | None = None,
        heading_sigma_deg: float | None = None,
    ) -> None:
        """Begin from a known position at the next frame, e.g. the last GNSS fix before
        jamming: ``xy`` in world coordinates, ``sigma_m`` how far off it may be.

        Without ``heading`` the compass gives it. Within reach of a local search the
        track starts at once (reported as coasting until matches support it);
        otherwise, or with no heading at all, it starts as a lost track's dead
        reckoning, whose region the acquisition searches.
        """
        self._start = (np.asarray(xy, float), float(sigma_m), heading, heading_sigma_deg)

    def _begin(self, frame: Frame) -> None:
        cfg, tel = self.cfg, frame.telemetry
        xy, sigma, heading, heading_sigma = self._start
        self._start = None
        if heading is None and tel.yaw is not None:
            heading = float(wrap_angle(tel.yaw + self.compass_offset))
            heading_sigma = heading_sigma or cfg.window.compass_sigma_deg
        known = heading is not None
        sh = np.radians(heading_sigma or cfg.meas_heading_sigma_deg) if known else np.pi
        pose = np.array([xy[0], xy[1], heading if known else 0.0])
        cov = np.diag([sigma**2, sigma**2, sh**2])
        w = PoseWindow(
            cfg.window, frame.t, tel, pose, cov, self.compass_offset, self.odometry_scale
        )
        self._started = (len(self.log) - 1, pose, cov)
        self.hypotheses, self._rejected, self._frames_in_state = [], [], 0
        if known and cfg.search_sigmas * sigma <= cfg.max_search_m:
            self.window, self.dead_reckoning = w, None
            self.status, self._window_start = Status.TRACKING, frame.t
        else:
            self.window, self.dead_reckoning = None, w
            self.status, self._lost_at, self._searches = Status.LOST, frame.t, 0

    def _step(self, frame: Frame, matchable: bool) -> Fix | None:
        """Track, acquire or converge on one frame; the match a track took, if any."""
        fix = None
        if self.dead_reckoning is not None:
            self._dead_reckon(frame)
        if self.status is Status.TRACKING:
            fix = self._track(frame, matchable)
        elif self.status is Status.LOST:
            # Consecutive frames overlap heavily, so searching every frame buys little.
            if matchable and self._frames_in_state % self.cfg.lost_search_every == 0:
                self._acquire(frame)
            if self.status is Status.LOST:
                self._frames_in_state += 1
        else:  # converging
            for h in self.hypotheses:
                self._predict(h, frame)
            if matchable:
                for h in self.hypotheses:
                    self._measure(h, frame)
            self.hypotheses = [
                h
                for h in self.hypotheses
                if frame.t - h.last_hit_t <= self.cfg.max_coast_converging_s
            ]
            self._frames_in_state += 1
            if matchable and self._frames_in_state % self.cfg.reacquire_every == 0:
                self._acquire(frame)
            self._try_confirm()
            if self.status is Status.CONVERGING and not self.hypotheses:
                self.status = Status.LOST
                self._frames_in_state = 0
        return fix

    # -- acquisition and confirmation -------------------------------------

    def _acquire(self, frame: Frame) -> None:
        cfg, dr = self.cfg, self.dead_reckoning
        yaw = frame.telemetry.yaw
        region = SearchRegion(
            center=cfg.prior_center,
            radius=cfg.prior_radius_m,
            heading=None if yaw is None else float(wrap_angle(yaw + self.compass_offset)),
            heading_span=np.radians(cfg.compass_prior_deg),
        )
        self._searches += 1
        fast = False  # whether a candidate inside the dead-reckoned region confirms faster
        if dr is not None:
            x, y, heading = dr.pose
            reach = cfg.search_sigmas * dr.sigma()
            fast = reach <= cfg.reacquire_radius_m and (
                frame.t - self._lost_at <= cfg.reacquire_window_s
            )
        if dr is not None and self._searches % cfg.wide_search_every:
            region = SearchRegion(
                center=(float(x), float(y)),
                # stay on the coarse (global) search path; it is restricted to this region
                radius=max(reach, cfg.matcher.global_radius_m + 1.0),
                heading=float(heading),
                heading_span=float(
                    np.clip(
                        cfg.search_sigmas * dr.heading_sigma(),
                        np.radians(cfg.max_heading_span_deg),
                        np.pi,
                    )
                ),
            )
        for f in self._match(frame, region):
            same = [h for h in self.hypotheses if self._same_place(h, f)]
            for h in same:
                h.distinct = max(h.distinct, f.distinctiveness)
            if same or f.score < cfg.init_min_score:
                continue
            self._candidate(f, frame.t, fast and bool(np.hypot(f.x - x, f.y - y) <= reach))

    def _candidate(self, f: Fix, t: float, near_track: bool) -> None:
        """Follow ``f`` (a match on the newest frame) as a candidate location."""
        cfg = self.cfg
        if len(self.hypotheses) >= cfg.max_hypotheses:
            return
        kf = PoseFilter(
            cfg.filter,
            f.x,
            f.y,
            f.heading,
            pos_sigma=cfg.meas_sigma_m,
            heading_sigma=np.radians(cfg.meas_heading_sigma_deg),
        )
        kf.P[:2, :2] += _spread(f)
        hyp = _Hypothesis(kf, [f.score], t, np.array([f.x, f.y]), 0.0, f.distinctiveness)
        hyp.frames, hyp.fixes = [len(self.log) - 1], {len(self.log) - 1: f}
        hyp.near_track = near_track
        self.hypotheses.append(hyp)
        if self.status is Status.LOST:
            self.status = Status.CONVERGING
            self._frames_in_state = 0

    def _same_place(self, h: _Hypothesis, f: Fix) -> bool:
        """Same position and heading; a heading flipped by ~180 degrees is a rival."""
        near = np.hypot(*(h.xy - (f.x, f.y))) < self.cfg.matcher.peak_separation_m
        return near and abs(wrap_angle(h.filter.x[4] - f.heading)) < np.pi / 2

    def _try_confirm(self) -> None:
        """Commit to the best hypothesis once it has clearly beaten every rival."""
        cfg = self.cfg
        if not self.hypotheses:
            return
        best = max(self.hypotheses, key=lambda h: h.mean_score)
        hits = sum(s > 0 for s in best.scores)
        if best.near_track:
            needed, distance = cfg.reacquire_confirm_frames, 0.0
        else:
            needed, distance = cfg.confirm_frames, cfg.confirm_distance_m
        if (
            hits < needed
            or best.travelled < distance
            or best.mean_score < cfg.init_min_score
            or best.distinct < cfg.init_min_distinctiveness
        ):
            return
        for other in self.hypotheses:
            # Compare over the frames both have seen, not over unequal histories.
            k = min(len(best.scores), len(other.scores))
            lead = np.mean(best.scores[-k:]) - np.mean(other.scores[-k:])
            if other is not best and lead < cfg.confirm_margin:
                return
        self._start_window(best)
        if self.takeoff_elevation is None and self.terrain is not None:
            self._learn_takeoff(best)
        self.hypotheses = []
        self.status = Status.TRACKING
        self.dead_reckoning = None

    def _learn_takeoff(self, h: _Hypothesis) -> None:
        """The takeoff's elevation from a confirmed candidate's matches.

        A match's scale is the height above the ground it saw, as a share of the
        reported height above takeoff; the ground's elevation is in the terrain.
        """
        guesses = []
        for i, f in h.fixes.items():
            tel = self.log[i]["tel"]
            if tel.above_takeoff and tel.altitude > 0:
                seen = f.scale * tel.altitude
                ahead = footprint(self.camera, seen, tel.pitch, tel.roll).center
                ground = self.terrain.at(np.array([f.x, f.y]) + rot2(f.heading) @ ahead)
                guesses.append(float(ground) + seen - tel.altitude)
        if guesses:
            self.takeoff_elevation = float(np.median(guesses))

    # -- per-hypothesis tracking ------------------------------------------

    def _predict(self, h: _Hypothesis, frame: Frame) -> None:
        cfg, kf = self.cfg, h.filter
        dt = frame.t - self._last_t if self._last_t is not None else 0.0
        yaw, last = frame.telemetry.yaw, self._last_yaw
        m = self._motion
        before = h.xy
        if m is not None:
            k = self._own_scale(h, frame.telemetry)
            kf.predict_motion(dt, k * m.d_body, m.d_heading, k * m.sigma_m, m.sigma_heading)
        else:
            kf.predict(dt, None if yaw is None or last is None else float(wrap_angle(yaw - last)))
        h.flown += float(np.hypot(*(h.xy - before)))
        if kf.pos_sigma > cfg.trust_sigma_m and h.trusted:
            h.trusted = False  # a wide search must be confirmed before it is trusted again
            h.streak, h.untrusted_since = 0, frame.t

    def _own_scale(self, h: _Hypothesis, used: Telemetry) -> float:
        """How much further ``h`` moved than odometry measured, at the height it flies.

        Odometry runs once per frame, at the height above the ground where the
        most likely position is (``used``). Ground motion grows with the height, so
        a candidate over lower ground moved further. On a real flight a wrong lead
        over high ground put every candidate at a quarter of its height above the
        ground, and the true one, moving at a quarter of its speed, kept dying.
        """
        if self._height is None or used.above_takeoff:
            return 1.0
        spread = min(h.filter.pos_sigma, self.cfg.max_search_m / self.cfg.search_sigmas)
        own = view_height(self._height, self.camera, used, h.xy, float(h.filter.x[4]), spread)
        return own / used.altitude if own > 0 and used.altitude > 0 else 1.0

    def _measure(self, h: _Hypothesis, frame: Frame) -> Fix | None:
        cfg, kf = self.cfg, h.filter
        region = SearchRegion(
            center=(float(kf.x[0]), float(kf.x[1])),
            radius=float(
                np.clip(cfg.search_sigmas * kf.pos_sigma, cfg.min_search_m, cfg.max_search_m)
            ),
            heading=float(kf.x[4]),
            heading_span=float(
                np.clip(
                    cfg.search_sigmas * kf.heading_sigma,
                    np.radians(cfg.min_heading_span_deg),
                    np.radians(cfg.max_heading_span_deg),
                )
            ),
        )
        fixes = self._match(frame, region)
        fix = fixes[0] if fixes else None
        min_excess = cfg.track_min_excess + (0.0 if h.trusted else cfg.untrusted_extra_excess)
        if fix is not None and (fix.score < cfg.track_min_score or fix.excess < min_excess):
            fix = None
        if fix is not None:
            sigma_h = np.radians(cfg.meas_heading_sigma_deg)
            R = np.diag([cfg.meas_sigma_m**2, cfg.meas_sigma_m**2, sigma_h**2])
            R[:2, :2] += _spread(fix)  # a ridge-shaped peak fixes position only across it
            if not h.trusted:
                R[:2, :2] *= cfg.untrusted_meas_inflation
            if not kf.update(np.array([fix.x, fix.y, fix.heading]), R):
                fix = None
        h.scores.append(fix.score if fix is not None else 0.0)
        if fix is None:
            h.streak = 0
            return None
        h.fixes[len(self.log) - 1] = fix
        if h.streak == 0:
            h.streak_from = h.flown
        h.streak += 1
        flown = h.flown - h.streak_from
        if h.streak >= cfg.reconfirm_frames and flown >= cfg.reconfirm_distance_m:
            h.trusted = True
        if h.last_hit_xy is not None:
            h.travelled += float(np.hypot(*(h.xy - h.last_hit_xy)))
        h.last_hit_t, h.last_hit_xy = frame.t, h.xy
        return fix

    # -- confirmed track: sliding window ------------------------------------

    def _start_window(self, h: _Hypothesis) -> None:
        """Replay the winning hypothesis's frames and matches into a window."""
        cfg = self.cfg
        first = h.frames[0]
        f0 = h.fixes[first]
        pose = np.array([f0.x, f0.y, f0.heading])
        cov = np.diag(
            [cfg.meas_sigma_m**2, cfg.meas_sigma_m**2, np.radians(cfg.meas_heading_sigma_deg) ** 2]
        )
        cov[:2, :2] += _spread(f0)
        log = self.log
        w = PoseWindow(
            cfg.window,
            log[first]["t"],
            log[first]["tel"],
            pose,
            cov,
            self.compass_offset,
            self.odometry_scale,
        )
        w.attach(f0)
        log[first]["fix"] = f0
        for i in h.frames[1:]:
            w.extend(log[i]["t"], log[i]["tel"], log[i]["motion"])
            if i in h.fixes:
                w.attach(h.fixes[i])
                log[i]["fix"] = h.fixes[i]
        w.solve()
        w.trim()
        self.window = w
        self._window_start, self._rejected = log[first]["t"], []

    def _track(self, frame: Frame, matchable: bool) -> Fix | None:
        cfg, w = self.cfg, self.window
        w.extend(frame.t, frame.telemetry, self._motion)
        w.solve()
        sigma = w.sigma()
        last = w.last_inlier_t() or self._window_start
        if cfg.search_sigmas * sigma > cfg.max_search_m or frame.t - last > cfg.max_coast_s:
            self._lose()
            return None
        fix = None
        if matchable:
            x, y, heading = w.pose
            region = SearchRegion(
                center=(float(x), float(y)),
                radius=float(
                    np.clip(cfg.search_sigmas * sigma, cfg.min_search_m, cfg.max_search_m)
                ),
                heading=float(heading),
                heading_span=float(
                    np.clip(
                        cfg.search_sigmas * w.heading_sigma(),
                        np.radians(cfg.min_heading_span_deg),
                        np.radians(cfg.max_heading_span_deg),
                    )
                ),
            )
            fixes = self._match(frame, region)
            fix = fixes[0] if fixes else None
            if fix is not None and (
                fix.score < cfg.track_min_score or fix.excess < cfg.track_min_excess
            ):
                fix = None
            if fix is not None:
                w.attach(fix)
                w.solve()
                self.log[-1]["fix"] = fix
                if w.nodes[-1].inlier:
                    self._learn(fix, frame)
                    self._rejected = []
                elif self._contradicted(frame.t, fix):
                    self._lose()
                    # the match that tipped it starts the place's candidacy
                    self._candidate(fix, frame.t, True)
                    return None
                else:
                    fix = None
        w.trim()
        self.compass_offset, self.odometry_scale = w.compass_offset, w.scale
        return fix

    def _trusted(self, w: PoseWindow) -> bool:
        cfg = self.cfg
        support = w.support(cfg.reconfirm_distance_m)
        return w.sigma() <= cfg.trust_sigma_m and support >= cfg.trust_support

    def _contradicted(self, t: float, fix: Fix) -> bool:
        """Whether matches the untrusted track rejects keep pointing to one other place."""
        cfg, w = self.cfg, self.window
        recent = [(s, o) for s, o in self._rejected if t - s <= cfg.contradiction_window_s]
        self._rejected = [*recent, (t, np.array([fix.x, fix.y]) - w.pose[:2])]
        if len(self._rejected) < cfg.contradiction_matches or self._trusted(w):
            return False
        offsets = np.array([o for _, o in self._rejected])
        spread = np.hypot(*(offsets - np.median(offsets, axis=0)).T)
        return bool(np.all(spread <= cfg.matcher.peak_separation_m / 2))

    def _lose(self) -> None:
        """Stop matching around the track, but keep dead reckoning it."""
        self.dead_reckoning, self.window = self.window, None
        self._lost_at, self._searches, self._rejected = self.dead_reckoning.nodes[-1].t, 0, []
        self.status = Status.LOST
        self._frames_in_state = 0

    def _dead_reckon(self, frame: Frame) -> None:
        w = self.dead_reckoning
        w.extend(frame.t, frame.telemetry, self._motion)
        w.solve()
        w.trim()
        if self.cfg.search_sigmas * w.sigma() > self.cfg.max_dead_reckoning_m:
            self.dead_reckoning = None

    def smooth(self) -> list[Estimate]:
        """Whole-flight solve over every frame since the first confirmed track, or the
        known position it began from (recorded footage).

        Frames before the first confirmation keep their causal estimates. A frame is
        trusted with a small smoothed uncertainty and enough independent inlier matches
        within half a window (span or frame count) of it.
        """
        cfg, log = self.cfg, self.log
        estimates = [e["estimate"] for e in log]
        start = next((i for i, e in enumerate(log) if e["fix"] is not None), None)
        if self._started is not None and (start is None or self._started[0] <= start):
            start, pose, cov = self._started  # from the known position
        elif start is None:
            return estimates
        else:
            f0 = log[start]["fix"]
            pose, cov = np.array([f0.x, f0.y, f0.heading]), np.diag([100.0, 100.0, 0.05])
        w = PoseWindow(
            replace(cfg.window, span_s=float("inf")),
            log[start]["t"],
            log[start]["tel"],
            pose,
            cov,
            self.compass_offset,
            self.odometry_scale,
        )
        if log[start]["fix"] is not None:
            w.attach(log[start]["fix"])
        for e in log[start + 1 :]:
            est = e["estimate"]
            guess = None  # start from the causal estimate where there is one
            if est.status is not Status.LOST and np.isfinite(est.x):
                guess = np.array([est.x, est.y, est.heading])
            w.extend(e["t"], e["tel"], e["motion"], guess)
            if e["fix"] is not None:
                w.attach(e["fix"])
        w.solve(covariances="all")
        steps = [np.hypot(*(b.pose[:2] - a.pose[:2])) for a, b in pairwise(w.nodes)]
        flown = np.concatenate([[0.0], np.cumsum(steps)])
        t = np.array([nd.t for nd in w.nodes])
        inlier = np.array([nd.fix is not None and nd.inlier for nd in w.nodes])
        half, frames = cfg.window.span_s / 2, cfg.window.min_frames // 2
        index = np.arange(len(w.nodes))
        for k, nd in enumerate(w.nodes):
            near = flown[inlier & ((np.abs(t - nd.t) <= half) | (np.abs(index - k) <= frames))]
            sigma = w.sigma(k)
            trusted = (
                sigma <= cfg.trust_sigma_m
                and _independent(near, cfg.reconfirm_distance_m) >= cfg.trust_support
            )
            old = estimates[start + k]
            if cfg.search_sigmas * sigma > cfg.max_dead_reckoning_m:  # as when live
                estimates[start + k] = Estimate(old.t, Status.LOST, latency_ms=old.latency_ms)
                continue
            estimates[start + k] = Estimate(
                old.t,
                Status.TRACKING if trusted else Status.COASTING,
                float(nd.pose[0]),
                float(nd.pose[1]),
                float(nd.pose[2]),
                sigma,
                old.fix,
                old.latency_ms,
            )
        return estimates

    # -- helpers ----------------------------------------------------------

    def _height_field(self, tel: Telemetry) -> HeightField | None:
        """The frame's height above the ground anywhere, if it is above takeoff over terrain.

        Where that would be below ``min_altitude_m``, the aircraft is not (NaN).
        """
        if not tel.above_takeoff or self.terrain is None or self.takeoff_elevation is None:
            return None
        above = self.terrain.height_above(self.takeoff_elevation + tel.altitude)
        floor = self.cfg.min_altitude_m

        def height(xy: np.ndarray) -> np.ndarray:
            h = above(xy)
            return np.where(h >= floor, h, np.nan)

        return height

    def _above_ground(self, frame: Frame, belief) -> Frame:
        """The frame with its height above the ground at the most likely position, if known.

        With no position yet, a height above takeoff is kept, unless the aircraft has
        left the ground and now reads too low to use: one that took off on a hill may
        read 0 m over a valley while 60 m above it. It then takes the typical height
        above the ground of where it may be.
        """
        tel = frame.telemetry
        if tel.above_takeoff and tel.altitude > ON_GROUND_M:
            self._airborne = True
        if belief is None:
            low = tel.altitude < self.cfg.min_altitude_m
            if self._height is None or not self._airborne or not low:
                return frame
            h = self._height(self._anywhere())
            if not np.isfinite(h).any():  # underground wherever it may be
                return frame
            height = float(np.nanmedian(h))
            return replace(frame, telemetry=replace(tel, altitude=height, above_takeoff=False))
        xy, heading, spread = belief
        height = view_height(self._height, self.camera, tel, xy, heading, spread)
        if not height > 0:  # the belief would put it underground
            return frame
        return replace(frame, telemetry=replace(tel, altitude=height, above_takeoff=False))

    def _anywhere(self) -> np.ndarray:
        """Points spread over where the aircraft may be: the prior region, or the map."""
        x0, y0, x1, y1 = self.map.extent
        if (c := self.cfg.prior_center) is not None and np.isfinite(r := self.cfg.prior_radius_m):
            x0, y0 = max(x0, c[0] - r), max(y0, c[1] - r)
            x1, y1 = min(x1, c[0] + r), min(y1, c[1] + r)
        gx, gy = np.meshgrid(np.linspace(x0, x1, 32), np.linspace(y0, y1, 32))
        return np.stack([gx, gy], -1).reshape(-1, 2)

    def _relief(self, frame: Frame, belief) -> Relief | None:
        """The terrain around the most likely position, for odometry to project onto."""
        tel = frame.telemetry
        if belief is None or tel.above_takeoff or not self.cfg.odometry.terrain_relief:
            return None
        (xy, heading, spread), height = belief, self._height
        turn = rot2(heading).T

        def relief(local: np.ndarray) -> np.ndarray:
            return relief_at(height, xy + local @ turn, tel.altitude, spread)

        return relief

    def _belief(self, t: float) -> tuple[np.ndarray, float, float] | None:
        """Most likely position at ``t``, heading, and how far off it may be (m).

        From the track, its dead reckoning or the lead candidate, carried forward
        at its last speed: the terrain is looked up where the aircraft is now.
        """
        cfg = self.cfg
        for w in (self.window, self.dead_reckoning):
            if w is not None:
                a, b = w.nodes[-2:] if len(w.nodes) > 1 else w.nodes[-1:] * 2
                speed = (b.pose[:2] - a.pose[:2]) / (b.t - a.t) if b.t > a.t else 0.0
                xy, heading, sigma = b.pose[:2] + speed * (t - b.t), float(b.pose[2]), w.sigma()
                break
        else:
            if not self.hypotheses:
                return None
            lead = max(self.hypotheses, key=lambda h: h.mean_score)
            xy, heading, sigma = lead.xy, float(lead.filter.x[4]), lead.filter.pos_sigma
        return xy, heading, min(sigma, cfg.max_search_m / cfg.search_sigmas)

    def _rolling(self, frame: Frame) -> bool:
        """Tilted beyond the rolling limit while the tilt changes fast (entering a bank)."""
        tel, cfg = frame.telemetry, self.cfg
        last = self.log[-1]["tel"] if self.log else None
        if last is None or tel.tilt <= np.radians(cfg.max_match_tilt_rolling_deg):
            return False
        dt = frame.t - self.log[-1]["t"]
        return dt <= 0 or abs(tel.tilt - last.tilt) / dt > np.radians(cfg.rolling_deg_s)

    def _plausible(self, m: Motion | None, t: float) -> Motion | None:
        """Odometry, unless it implies an impossible speed (RANSAC on a wrong cluster)."""
        if m is None or self._last_t is None:
            return m
        dt = t - self._last_t
        return m if np.hypot(*m.d_body) <= self.cfg.max_speed * dt else None

    def _match(self, frame: Frame, region: SearchRegion) -> list[Fix]:
        try:
            return self.matcher.match(frame, region, self.scale, self._height)
        except ValueError:  # e.g. camera tilted above the horizon
            return []

    def _learn(self, fix: Fix, frame: Frame) -> None:
        """Slowly track altitude bias and compass offset from confirmed fixes."""
        a = self.cfg.smoothing
        self.scale += a * (fix.scale - self.scale)
        if frame.telemetry.yaw is not None:
            err = float(wrap_angle(fix.heading - frame.telemetry.yaw - self.compass_offset))
            self.compass_offset = float(wrap_angle(self.compass_offset + a * err))

    def _estimate(self, t: float, fix: Fix | None, latency_ms: float) -> Estimate:
        estimate = self._current(t, fix, latency_ms)
        self.log[-1]["estimate"] = estimate
        return estimate

    def _current(self, t: float, fix: Fix | None, latency_ms: float) -> Estimate:
        if self.status is Status.TRACKING and self.window is not None:
            w = self.window
            x, y, heading = w.pose
            status = Status.TRACKING if self._trusted(w) else Status.COASTING
            return Estimate(
                t, status, float(x), float(y), float(heading), w.sigma(), fix, latency_ms
            )
        if self.dead_reckoning is not None:
            w = self.dead_reckoning
            x, y, heading = w.pose
            return Estimate(
                t, Status.COASTING, float(x), float(y), float(heading), w.sigma(), fix, latency_ms
            )
        if not self.hypotheses:
            return Estimate(t, Status.LOST, latency_ms=latency_ms)
        lead = max(self.hypotheses, key=lambda h: h.mean_score)
        x, y, _, _, heading = lead.filter.x
        return Estimate(
            t,
            self.status,
            float(x),
            float(y),
            float(heading),
            lead.filter.pos_sigma,
            fix,
            latency_ms,
        )


def _spread(fix: Fix) -> np.ndarray:
    ee, en, nn = fix.spread
    return np.array([[ee, en], [en, nn]])


def _independent(flown: np.ndarray, separation_m: float) -> int:
    """Matches at these flown distances, counting those within ``separation_m`` of a run once."""
    count, last = 0, -np.inf
    for f in np.sort(flown):
        if f - last >= separation_m:
            count, last = count + 1, f
    return count
