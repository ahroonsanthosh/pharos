"""Frame-by-frame absolute localization: matcher + filters + a small state machine.

    LOST ──(global fixes)──▶ CONVERGING ──(one hypothesis wins)──▶ TRACKING
     ▲                          │                                    │
     └──────(all rejected)──────┴───────(no match for too long)───────┘

A wrong-but-confident position is worse than admitting the aircraft is lost.
Consecutive frames overlap almost completely, so a look-alike place that fools
one frame fools the next few as well. Instead of trusting the best global fix,
the localizer keeps several candidate locations, follows each one with local
searches as the aircraft flies over new ground, and only commits when one
candidate has clearly out-scored the others over a minimum distance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from pharos.geomap import GeoMap
from pharos.geometry import Camera, wrap_angle
from pharos.nav.filter import FilterConfig, PoseFilter
from pharos.types import Estimate, Fix, Frame, Status, Telemetry
from pharos.vision.matcher import CorrelationMatcher, Matcher, MatcherConfig, SearchRegion
from pharos.vision.odometry import Motion, OdometryConfig, VisualOdometry


@dataclass(frozen=True)
class LocalizerConfig:
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    odometry: OdometryConfig | None = field(default_factory=OdometryConfig)  # None: off
    # Frames from lower or with the camera tilted further from straight down are skipped:
    # on the ground, taking off, or looking at the horizon.
    min_altitude_m: float = 15.0
    max_tilt_deg: float = 60.0
    # Acquisition
    prior_center: tuple[float, float] | None = None  # e.g. last GNSS fix before denial
    prior_radius_m: float = float("inf")
    compass_prior_deg: float = 45.0  # heading search half-width around the compass when lost
    init_min_score: float = 0.25
    init_min_distinctiveness: float = 0.1  # a winner must once have stood out map-wide
    max_hypotheses: int = 4
    lost_search_every: int = 2  # frames between global searches while lost
    reacquire_every: int = 6  # frames between global searches while converging
    # After losing a confirmed track, search where the aircraft can have flown since,
    # and confirm a candidate there with fewer matches (continuity is evidence too).
    reacquire_window_s: float = 60.0
    max_speed: float = 30.0  # m/s
    reacquire_confirm_frames: int = 3  # matches including the acquisition (vs confirm_frames)
    # Confirmation
    confirm_frames: int = 4
    confirm_distance_m: float = 120.0  # ground covered before a candidate can win
    confirm_margin: float = 0.06  # mean-score lead over the runner-up
    # Tracking
    track_min_score: float = 0.18
    track_min_excess: float = 1.5  # a match must stand out from its window's noise
    max_coast_converging_s: float = 5.0  # a candidate dies after this long without a match
    # A confirmed track keeps searching around its prediction (reported as coasting) while
    # the window that needs stays within max_search_m, for up to max_coast_s.
    max_coast_s: float = 30.0
    trust_sigma_m: float = 15.0  # beyond this uncertainty a track is reported as coasting
    reconfirm_frames: int = 2  # consecutive matches needed to trust a track again after coasting
    meas_sigma_m: float = 3.0
    meas_heading_sigma_deg: float = 2.0
    search_sigmas: float = 3.0
    min_search_m: float = 15.0
    max_search_m: float = 150.0
    min_heading_span_deg: float = 6.0  # at least three fine heading steps
    max_heading_span_deg: float = 30.0
    smoothing: float = 0.2  # EMA weight for altitude-scale and compass-offset estimates
    odometry_gain_range: tuple[float, float] = (0.7, 4.0)  # bounds on learned odometry noise


def usable_view(tel: Telemetry, cfg: LocalizerConfig) -> bool:
    """Whether a frame can be matched: high enough, and not looking at the horizon."""
    return tel.altitude >= cfg.min_altitude_m and tel.tilt <= np.radians(cfg.max_tilt_deg)


@dataclass
class _Hypothesis:
    filter: PoseFilter
    scores: list[float] = field(default_factory=list)
    last_hit_t: float = 0.0
    last_hit_xy: np.ndarray | None = None
    travelled: float = 0.0  # distance between matched positions, not dead reckoning
    distinct: float = 0.0  # best map-wide distinctiveness seen when (re)acquired here
    trusted: bool = True  # False after coasting, until re-confirmed by consecutive matches
    near_lost_track: bool = False  # found where a just-lost track could have flown to
    streak: int = 0  # consecutive matches

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
    ):
        self.cfg = cfg or LocalizerConfig()
        self.map = geomap
        self.matcher = matcher or CorrelationMatcher(geomap, camera, self.cfg.matcher)
        odometry = self.cfg.odometry
        self.odometry = VisualOdometry(camera, odometry) if odometry is not None else None
        self._motion: Motion | None = None
        self.status = Status.LOST
        self.hypotheses: list[_Hypothesis] = []
        self.scale = 1.0  # running estimate of true / reported altitude
        # Odometry error varies by platform (a banking fixed-wing vs a stabilized
        # gimbal), so confirmed tracks learn a factor on its stated uncertainty.
        self.odometry_gain = 1.0
        self._odometry_nis = 2.0  # running normalized innovation; 2 when the model fits
        self.compass_offset = 0.0  # running estimate of true heading - compass
        self._frames_in_state = 0
        self._lost_track: tuple[float, np.ndarray] | None = None  # (time, xy) of its last match
        self._last_t: float | None = None
        self._last_yaw: float | None = None

    def process(self, frame: Frame) -> Estimate:
        start = time.perf_counter()
        fix = None
        usable = usable_view(frame.telemetry, self.cfg)
        self._motion = None
        if self.odometry is not None:
            if usable:
                self._motion = self.odometry.step(frame, self.scale)
            else:
                self.odometry.reset()
        if self.status is Status.LOST:
            # Consecutive frames overlap heavily, so searching every frame buys little.
            if usable and self._frames_in_state % self.cfg.lost_search_every == 0:
                self._acquire(frame)
            if self.status is Status.LOST:
                self._frames_in_state += 1
        else:
            for h in self.hypotheses:
                self._predict(h, frame)
            if self.status is Status.TRACKING:
                # Decide before searching: a wide search must not resurrect a lost track.
                for h in self.hypotheses:
                    if not self._searchable(h, frame.t) and h.last_hit_xy is not None:
                        self._lost_track = (h.last_hit_t, h.last_hit_xy.copy())
                self.hypotheses = [h for h in self.hypotheses if self._searchable(h, frame.t)]
            fixes = [self._measure(h, frame) if usable else None for h in self.hypotheses]
            alive = [
                (h, f)
                for h, f in zip(self.hypotheses, fixes, strict=True)
                if self.status is Status.TRACKING
                or frame.t - h.last_hit_t <= self.cfg.max_coast_converging_s
            ]
            self.hypotheses = [h for h, _ in alive]
            if self.status is Status.TRACKING and alive:
                fix = alive[0][1]
                if fix is not None and alive[0][0].trusted:
                    self._learn(fix, frame)
            elif self.status is Status.CONVERGING:
                self._frames_in_state += 1
                if usable and self._frames_in_state % self.cfg.reacquire_every == 0:
                    self._acquire(frame)
                self._try_confirm()
            if not self.hypotheses:
                self.status = Status.LOST
                self._frames_in_state = 0
        self._last_t = frame.t
        self._last_yaw = frame.telemetry.yaw
        return self._estimate(frame.t, fix, (time.perf_counter() - start) * 1e3)

    # -- acquisition and confirmation -------------------------------------

    def _acquire(self, frame: Frame) -> None:
        cfg = self.cfg
        yaw = frame.telemetry.yaw
        center, radius = cfg.prior_center, cfg.prior_radius_m
        reach = self._reach(frame.t)
        if reach is not None:
            center, radius = tuple(self._lost_track[1]), reach
            # stay on the coarse (global) search path; it is restricted to this region
            radius = max(radius, cfg.matcher.global_radius_m + 1.0)
        region = SearchRegion(
            center=center,
            radius=radius,
            heading=None if yaw is None else float(wrap_angle(yaw + self.compass_offset)),
            heading_span=np.radians(cfg.compass_prior_deg),
        )
        for f in self._match(frame, region):
            same = [h for h in self.hypotheses if self._same_place(h, f)]
            for h in same:
                h.distinct = max(h.distinct, f.distinctiveness)
            if same or f.score < cfg.init_min_score or len(self.hypotheses) >= cfg.max_hypotheses:
                continue
            kf = PoseFilter(
                cfg.filter,
                f.x,
                f.y,
                f.heading,
                pos_sigma=cfg.meas_sigma_m,
                heading_sigma=np.radians(cfg.meas_heading_sigma_deg),
            )
            kf.P[:2, :2] += _spread(f)
            hyp = _Hypothesis(kf, [f.score], frame.t, np.array([f.x, f.y]), 0.0, f.distinctiveness)
            if reach is not None:
                hyp.near_lost_track = bool(np.hypot(*(hyp.xy - self._lost_track[1])) <= reach)
            self.hypotheses.append(hyp)
        if self.hypotheses and self.status is Status.LOST:
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
        if best.near_lost_track:
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
        self.hypotheses = [best]
        self.status = Status.TRACKING
        self._lost_track = None

    # -- per-hypothesis tracking ------------------------------------------

    def _predict(self, h: _Hypothesis, frame: Frame) -> None:
        cfg, kf = self.cfg, h.filter
        dt = frame.t - self._last_t if self._last_t is not None else 0.0
        yaw, last = frame.telemetry.yaw, self._last_yaw
        m = self._motion
        if m is not None:
            g = self.odometry_gain
            kf.predict_motion(dt, m.d_body, m.d_heading, g * m.sigma_m, g * m.sigma_heading)
        else:
            kf.predict(dt, None if yaw is None or last is None else float(wrap_angle(yaw - last)))
        if kf.pos_sigma > cfg.trust_sigma_m:
            h.trusted = False  # a wide search must be confirmed before it is trusted again

    def _measure(self, h: _Hypothesis, frame: Frame) -> Fix | None:
        cfg, kf, m = self.cfg, h.filter, self._motion
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
        if fix is not None and (
            fix.score < cfg.track_min_score or fix.excess < cfg.track_min_excess
        ):
            fix = None
        if fix is not None:
            sigma_h = np.radians(cfg.meas_heading_sigma_deg)
            R = np.diag([cfg.meas_sigma_m**2, cfg.meas_sigma_m**2, sigma_h**2])
            R[:2, :2] += _spread(fix)  # a ridge-shaped peak fixes position only across it
            z = np.array([fix.x, fix.y, fix.heading])
            nu, S, _ = kf.innovation(z, R)
            if not kf.update(z, R):
                fix = None
            elif m is not None and self.status is Status.TRACKING and h.trusted:
                self._learn_odometry(float(nu[:2] @ np.linalg.solve(S[:2, :2], nu[:2])))
        h.scores.append(fix.score if fix is not None else 0.0)
        h.streak = h.streak + 1 if fix is not None else 0
        if not h.trusted and h.streak >= cfg.reconfirm_frames:
            h.trusted = True
        if fix is not None:
            if h.last_hit_xy is not None:
                h.travelled += float(np.hypot(*(h.xy - h.last_hit_xy)))
            h.last_hit_t, h.last_hit_xy = frame.t, h.xy
        return fix

    # -- helpers ----------------------------------------------------------

    def _reach(self, t: float) -> float | None:
        """How far the aircraft can be from the last match of a recently lost track."""
        if self._lost_track is None:
            return None
        elapsed = t - self._lost_track[0]
        if elapsed > self.cfg.reacquire_window_s:
            self._lost_track = None
            return None
        return self.cfg.max_speed * elapsed + 50.0

    def _searchable(self, h: _Hypothesis, t: float) -> bool:
        """A confirmed track lives while its predicted position can still be searched for."""
        cfg = self.cfg
        within = cfg.search_sigmas * h.filter.pos_sigma <= cfg.max_search_m
        return within and t - h.last_hit_t <= cfg.max_coast_s

    def _match(self, frame: Frame, region: SearchRegion) -> list[Fix]:
        try:
            return self.matcher.match(frame, region, self.scale)
        except ValueError:  # e.g. camera tilted above the horizon
            return []

    def _learn_odometry(self, nis: float) -> None:
        """Scale odometry's uncertainty so position innovations look as expected (chi2, 2 dof)."""
        a = self.cfg.smoothing
        self._odometry_nis += a * (min(nis, 20.0) - self._odometry_nis)
        lo, hi = self.cfg.odometry_gain_range
        target = self.odometry_gain * np.sqrt(self._odometry_nis / 2.0)
        self.odometry_gain = float(
            np.clip(self.odometry_gain + a * (target - self.odometry_gain), lo, hi)
        )

    def _learn(self, fix: Fix, frame: Frame) -> None:
        """Slowly track altitude bias and compass offset from confirmed fixes."""
        a = self.cfg.smoothing
        self.scale += a * (fix.scale - self.scale)
        if frame.telemetry.yaw is not None:
            err = float(wrap_angle(fix.heading - frame.telemetry.yaw - self.compass_offset))
            self.compass_offset = float(wrap_angle(self.compass_offset + a * err))

    def _estimate(self, t: float, fix: Fix | None, latency_ms: float) -> Estimate:
        if not self.hypotheses:
            return Estimate(t, Status.LOST, latency_ms=latency_ms)
        lead = max(self.hypotheses, key=lambda h: h.mean_score)
        x, y, _, _, heading = lead.filter.x
        status = self.status
        if status is Status.TRACKING and (
            not lead.trusted or lead.filter.pos_sigma > self.cfg.trust_sigma_m
        ):
            status = Status.COASTING
        return Estimate(
            t,
            status,
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
