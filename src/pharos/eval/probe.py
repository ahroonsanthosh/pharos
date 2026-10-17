"""How well does each frame of real footage identify its own location?

For every frame that carries the aircraft's GNSS, the tracking search runs
twice: in a window around the GNSS position, and in a decoy window of the same
size a few hundred metres away. A good matcher finds a peak that clearly stands
out (high excess) near the truth and nothing in the decoy. Pass rates at an
excess threshold are the metric for tuning on real imagery: they need no flight
dynamics, so a few dozen frames compare configurations fairly.

The confident true-window matches also calibrate the camera. Searched again
over a wide range of scales, their offset from GNSS, seen from the camera, is
the mount misalignment (plus any constant video delay), and their scale is the
error of height times lens field of view.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import numpy as np

from pharos.footage import Footage
from pharos.geomap import GeoMap
from pharos.geometry import rot2
from pharos.nav.localizer import LocalizerConfig, usable_view
from pharos.vision.matcher import CorrelationMatcher, SearchRegion

THRESHOLDS = (1.0, 1.5, 2.0)


@dataclass(frozen=True)
class ProbeConfig:
    radius_m: float = 40.0  # search window half-width, like a tracking search
    jitter_m: float = 15.0  # the true window is centred up to this far from the GNSS fix
    decoy_m: float = 300.0  # distance of the decoy window
    heading_span_deg: float = 10.0  # around the reported yaw; all headings if there is none
    correct_m: float = 20.0  # a true-window peak within this of GNSS counts as correct
    calibrate_excess: float = 1.5  # matches used for calibration must stand out this much
    calibrate_scales: tuple[float, ...] = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25)


def probe(
    footage: Footage,
    geomap: GeoMap,
    config: LocalizerConfig | None = None,
    cfg: ProbeConfig | None = None,
    every: int = 1,
    seed: int = 0,
    progress=None,
) -> list[dict]:
    cfg = cfg or ProbeConfig()
    config = config or LocalizerConfig()
    rng = np.random.default_rng(seed)
    camera = footage.recording.camera
    m = CorrelationMatcher(geomap, camera, config.matcher)
    wide = CorrelationMatcher(geomap, camera, replace(config.matcher, scales=cfg.calibrate_scales))
    rows = []
    n = -(-len(footage.recording) // every)
    frames = itertools.islice(zip(footage.recording, footage.gnss, strict=True), 0, None, every)
    for i, (frame, gnss) in enumerate(frames):
        if gnss is None or (xy := geomap.from_latlon(*gnss)) is None:
            continue
        if not usable_view(frame.telemetry, config):
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
            "altitude_m": frame.telemetry.altitude,
        }
        if true and true.excess >= cfg.calibrate_excess and row["true_error_m"] <= cfg.correct_m:
            fine = _best(wide, frame, (true.x, true.y), 10.0, true.heading, math.radians(4))
            if fine:
                # (right, forward) from the GNSS position to the match, in the camera's frame
                offset = rot2(fine.heading).T @ (fine.x - xy[0], fine.y - xy[1])
                half = math.atan(fine.scale * camera.width / (2 * camera.fx))
                row |= {
                    "calibration_offset_m": offset.round(2).tolist(),
                    "scale": fine.scale,
                    "hfov_deg": math.degrees(2 * half),  # the same scale, as a lens correction
                }
        rows.append(row)
        if progress:
            progress(i + 1, n, row)
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
    good = [r for r in rows if "calibration_offset_m" in r]
    if good:
        right, forward = np.median([r["calibration_offset_m"] for r in good], axis=0)
        height = float(np.median([r["altitude_m"] * r["scale"] for r in good]))
        out["calibration"] = {
            "frames": len(good),
            # tilt the camera by this much more (forward, right) to cancel the offset
            "mount_offset_deg": [
                round(float(np.degrees(np.arctan2(forward, height))), 2),
                round(float(np.degrees(np.arctan2(right, height))), 2),
            ],
            # true / reported height, or equally tan(true / assumed half field of view)
            "altitude_scale": round(float(np.median([r["scale"] for r in good])), 3),
            "hfov_deg": round(float(np.median([r["hfov_deg"] for r in good])), 1),
        }
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
