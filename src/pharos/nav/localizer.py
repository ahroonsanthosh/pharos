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
from pharos.types import Estimate, Fix, Frame, Status
from pharos.vision.matcher import CorrelationMatcher, Matcher, MatcherConfig, SearchRegion


@dataclass(frozen=True)
class LocalizerConfig:
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    # Acquisition
    prior_center: tuple[float, float] | None = None  # e.g. last GNSS fix before denial
    prior_radius_m: float = float("inf")
    compass_prior_deg: float = 45.0  # heading search half-width around the compass when lost
    init_min_score: float = 0.25
    init_min_distinctiveness: float = 0.1  # a winner must once have stood out map-wide
    max_hypotheses: int = 4
    lost_search_every: int = 2  # frames between global searches while lost
    reacquire_every: int = 6  # frames between global searches while converging
    # Confirmation
    confirm_frames: int = 4
    confirm_distance_m: float = 120.0  # ground covered before a candidate can win
    confirm_margin: float = 0.06  # mean-score lead over the runner-up
    # Tracking
    track_min_score: float = 0.18
    track_min_excess: float = 1.5  # a match must stand out from its window's noise
    max_coast_converging_s: float = 5.0  # a candidate dies after this long without a match
    max_coast_s: float = 8.0  # a confirmed track is dropped after this long without a match
    trust_sigma_m: float = 15.0  # beyond this uncertainty a track is reported as coasting
    meas_sigma_m: float = 3.0
    meas_heading_sigma_deg: float = 2.0
    search_sigmas: float = 3.0
    min_search_m: float = 15.0
    max_search_m: float = 150.0
    min_heading_span_deg: float = 6.0  # at least three fine heading steps
    max_heading_span_deg: float = 30.0
    smoothing: float = 0.2  # EMA weight for altitude-scale and compass-offset estimates


@dataclass
class _Hypothesis:
    filter: PoseFilter
    scores: list[float] = field(default_factory=list)
    last_hit_t: float = 0.0
    last_hit_xy: np.ndarray | None = None
    travelled: float = 0.0  # distance between matched positions, not dead reckoning
    distinct: float = 0.0  # best map-wide distinctiveness seen when (re)acquired here

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
        self.status = Status.LOST
        self.hypotheses: list[_Hypothesis] = []
        self.scale = 1.0  # running estimate of true / reported altitude
        self.compass_offset = 0.0  # running estimate of true heading - compass
        self._frames_in_state = 0
        self._last_t: float | None = None
        self._last_yaw: float | None = None

    def process(self, frame: Frame) -> Estimate:
        start = time.perf_counter()
        fix = None
        if self.status is Status.LOST:
            # Consecutive frames overlap heavily, so searching every frame buys little.
            if self._frames_in_state % self.cfg.lost_search_every == 0:
                self._acquire(frame)
            if self.status is Status.LOST:
                self._frames_in_state += 1
        else:
            fixes = [self._step(h, frame) for h in self.hypotheses]
            limit = (
                self.cfg.max_coast_s
                if self.status is Status.TRACKING
                else self.cfg.max_coast_converging_s
            )
            alive = [
                (h, f)
                for h, f in zip(self.hypotheses, fixes, strict=True)
                if frame.t - h.last_hit_t <= limit
            ]
            self.hypotheses = [h for h, _ in alive]
            if self.status is Status.TRACKING and alive:
                fix = alive[0][1]
                if fix is not None:
                    self._learn(fix, frame)
            elif self.status is Status.CONVERGING:
                self._frames_in_state += 1
                if self._frames_in_state % self.cfg.reacquire_every == 0:
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
        region = SearchRegion(
            center=cfg.prior_center,
            radius=cfg.prior_radius_m,
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
            hyp = _Hypothesis(kf, [f.score], frame.t, np.array([f.x, f.y]), 0.0, f.distinctiveness)
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
        if (
            hits < cfg.confirm_frames
            or best.travelled < cfg.confirm_distance_m
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

    # -- per-hypothesis tracking ------------------------------------------

    def _step(self, h: _Hypothesis, frame: Frame) -> Fix | None:
        cfg, kf = self.cfg, h.filter
        dt = frame.t - self._last_t if self._last_t is not None else 0.0
        yaw, last = frame.telemetry.yaw, self._last_yaw
        kf.predict(dt, None if yaw is None or last is None else float(wrap_angle(yaw - last)))

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
            if not kf.update(np.array([fix.x, fix.y, fix.heading]), R):
                fix = None
        h.scores.append(fix.score if fix is not None else 0.0)
        if fix is not None:
            if h.last_hit_xy is not None:
                h.travelled += float(np.hypot(*(h.xy - h.last_hit_xy)))
            h.last_hit_t, h.last_hit_xy = frame.t, h.xy
        return fix

    # -- helpers ----------------------------------------------------------

    def _match(self, frame: Frame, region: SearchRegion) -> list[Fix]:
        try:
            return self.matcher.match(frame, region, self.scale)
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
        if not self.hypotheses:
            return Estimate(t, Status.LOST, latency_ms=latency_ms)
        lead = max(self.hypotheses, key=lambda h: h.mean_score)
        x, y, _, _, heading = lead.filter.x
        status = self.status
        if status is Status.TRACKING and lead.filter.pos_sigma > self.cfg.trust_sigma_m:
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
