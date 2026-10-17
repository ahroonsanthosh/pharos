"""How well does each frame of real footage identify its own location?

For every frame that carries the aircraft's GNSS, the tracking search runs
twice: in a window around the GNSS position, and in a decoy window of the same
size a few hundred metres away. A good matcher finds a peak that clearly stands
out (high excess) near the truth and nothing in the decoy. Pass rates at an
excess threshold are the metric for tuning on real imagery: they need no flight
dynamics, so a few dozen frames compare configurations fairly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pharos.footage import Footage
from pharos.geomap import GeoMap
from pharos.vision.matcher import CorrelationMatcher, MatcherConfig, SearchRegion

THRESHOLDS = (1.0, 1.5, 2.0)


@dataclass(frozen=True)
class ProbeConfig:
    radius_m: float = 40.0  # search window half-width, like a tracking search
    jitter_m: float = 15.0  # the true window is centred up to this far from the GNSS fix
    decoy_m: float = 300.0  # distance of the decoy window
    heading_span_deg: float = 10.0  # around the reported yaw; all headings if there is none
    correct_m: float = 20.0  # a true-window peak within this of GNSS counts as correct


def probe(
    footage: Footage,
    geomap: GeoMap,
    matcher: MatcherConfig | None = None,
    cfg: ProbeConfig | None = None,
    every: int = 1,
    seed: int = 0,
    progress=None,
) -> list[dict]:
    cfg = cfg or ProbeConfig()
    rng = np.random.default_rng(seed)
    m = CorrelationMatcher(geomap, footage.recording.camera, matcher)
    rows = []
    frames = list(zip(footage.recording, footage.gnss, strict=True))[::every]
    for i, (frame, gnss) in enumerate(frames):
        if gnss is None or (xy := geomap.from_latlon(*gnss)) is None:
            continue
        yaw = frame.telemetry.yaw
        span = math.radians(cfg.heading_span_deg) if yaw is not None else math.pi
        jitter = rng.uniform(-cfg.jitter_m, cfg.jitter_m, 2)
        true = _best(m, frame, (xy[0] + jitter[0], xy[1] + jitter[1]), cfg.radius_m, yaw, span)
        decoy_xy = _decoy(geomap, xy, cfg, rng)
        decoy = _best(m, frame, decoy_xy, cfg.radius_m, yaw, span) if decoy_xy else None
        row = {
            "t": frame.t,
            "true_excess": true.excess if true else None,
            "true_score": true.score if true else None,
            "true_error_m": math.hypot(true.x - xy[0], true.y - xy[1]) if true else None,
            "decoy_excess": decoy.excess if decoy else None,
            "decoy_score": decoy.score if decoy else None,
        }
        rows.append(row)
        if progress:
            progress(i + 1, len(frames), row)
    return rows


def summarize_probe(rows: list[dict], cfg: ProbeConfig | None = None) -> dict:
    cfg = cfg or ProbeConfig()
    true = np.array([r["true_excess"] if r["true_excess"] is not None else -9 for r in rows])
    near = np.array([(r["true_error_m"] or 1e9) <= cfg.correct_m for r in rows])
    decoy = np.array([r["decoy_excess"] for r in rows if r["decoy_excess"] is not None])
    out: dict = {
        "frames": len(rows),
        "true_excess_median": float(np.median(true)) if len(true) else None,
        "decoy_excess_median": float(np.median(decoy)) if len(decoy) else None,
        "true_correct": float(near.mean()) if len(near) else None,
    }
    for thr in THRESHOLDS:
        out[f"true_pass_{thr}"] = float(np.mean((true >= thr) & near)) if len(true) else None
        out[f"decoy_pass_{thr}"] = float(np.mean(decoy >= thr)) if len(decoy) else None
    return out


def _best(m: CorrelationMatcher, frame, center, radius, yaw, span):
    fixes = m.match(frame, SearchRegion(center, radius, yaw, span))
    return fixes[0] if fixes else None


def _decoy(geomap: GeoMap, xy, cfg: ProbeConfig, rng) -> tuple[float, float] | None:
    for _ in range(20):
        a = rng.uniform(0, 2 * math.pi)
        x, y = xy[0] + cfg.decoy_m * math.cos(a), xy[1] + cfg.decoy_m * math.sin(a)
        if geomap.contains(x, y, margin=cfg.radius_m + 50):
            return x, y
    return None
