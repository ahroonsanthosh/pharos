"""Find where a video sits in its flight log, from the footage itself.

A separate camera and autopilot share no clock. With the wrong offset every
frame is paired with the airframe attitude of another moment, so while the
aircraft rolls and pitches, the ground under each frame is placed wrongly and
visual odometry measures a jerky path. Aircraft do not fly jerkily: the offset
at which consecutive odometry steps agree best is the sync. No GNSS needed,
but the stretch of flight must contain attitude changes (a few turns, or the
wobble of a fixed-wing); steady straight flight or a stabilized gimbal leaves
no signal, which a low ``contrast`` reports.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from pharos.geometry import Camera, rot2
from pharos.nav.localizer import LocalizerConfig, usable_view
from pharos.telemetry import TelemetrySource
from pharos.types import Frame
from pharos.vision.odometry import OdometryConfig, VisualOdometry

_VIEW = LocalizerConfig()  # frames the localizer would skip are skipped here too


@dataclass(frozen=True)
class SyncResult:
    offset_s: float  # telemetry for video time t is at telemetry time t + offset_s
    jitter_m: float  # median change between consecutive odometry steps at that offset
    initial_jitter_m: float  # the same at the pairing given (offset 0)
    contrast: float  # 1 - best / median jitter; below ~0.15 the minimum is not trustworthy
    curve: list[tuple[float, float]]  # (offset, jitter) for every offset tried


def estimate_sync(
    frames: Iterable[Frame],
    telemetry: TelemetrySource,
    camera: Camera,
    span_s: float = 2.0,
    step_s: float = 0.1,
    fine_step_s: float = 0.02,
    odometry: OdometryConfig | None = None,
) -> SyncResult:
    """Best offset within ``±span_s`` of the current pairing of ``frames`` and ``telemetry``.

    ``frames`` should cover a minute or more of flight with some manoeuvring;
    only their images and times are used.
    """
    vo = VisualOdometry(camera, odometry)
    times, pairs, prev = [], [], None
    for frame in frames:
        features = vo.features(frame.image)
        if prev is not None:
            times.append((prev[0], frame.t))
            pairs.append(vo.correspondences(prev[1], features))
        prev = (frame.t, features)

    def jitter(offset: float) -> float:
        steps = []
        for (ta, tb), match in zip(times, pairs, strict=True):
            a, b = telemetry.at(ta + offset), telemetry.at(tb + offset)
            motion = None
            if match is not None and a is not None and b is not None and _usable(a, b):
                motion = vo.motion(*match, a.telemetry, b.telemetry)
            heading = a.telemetry.yaw if a is not None else None
            # world-frame steps, so turning at a steady rate does not count as jitter
            steps.append(None if motion is None else rot2(heading or 0.0) @ motion.d_body)
        changes = [
            np.hypot(*(q - p)) for p, q in pairwise(steps) if p is not None and q is not None
        ]
        return float(np.median(changes)) if len(changes) >= 10 else float("inf")

    def sweep(center: float, half: float, step: float) -> dict[float, float]:
        offsets = np.round(np.arange(center - half, center + half + step / 2, step), 3)
        return {float(o): jitter(o) for o in offsets}

    curve = sweep(0.0, span_s, step_s)
    curve |= sweep(min(curve, key=curve.get), step_s, fine_step_s)
    offset = min(curve, key=curve.get)
    if not np.isfinite(curve[offset]):
        raise ValueError("too few overlapping frames with telemetry to estimate the sync")
    finite = [v for v in curve.values() if np.isfinite(v)]
    contrast = 1.0 - curve[offset] / float(np.median(finite))
    initial = curve.get(0.0, jitter(0.0))
    return SyncResult(offset, curve[offset], initial, contrast, sorted(curve.items()))


def _usable(a, b) -> bool:
    return usable_view(a.telemetry, _VIEW) and usable_view(b.telemetry, _VIEW)
