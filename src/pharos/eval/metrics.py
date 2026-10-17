"""Accuracy, availability and integrity metrics for a localized flight."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pharos.geometry import wrap_angle
from pharos.types import Estimate, GroundTruth, Status

WRONG_M = 25.0  # a TRACKING estimate further off than this counts as confidently wrong


def evaluate(estimates: Sequence[Estimate], truth: Sequence[GroundTruth]) -> dict[str, float]:
    if len(estimates) != len(truth):
        raise ValueError("estimates and truth must align frame by frame")
    n = len(truth)
    err = np.array([np.hypot(e.x - g.x, e.y - g.y) for e, g in zip(estimates, truth, strict=True)])
    herr = np.degrees(
        np.abs([wrap_angle(e.heading - g.heading) for e, g in zip(estimates, truth, strict=True)])
    )
    trk = np.array([e.status is Status.TRACKING for e in estimates], bool)
    first = np.flatnonzero(trk)
    latency = np.array([e.latency_ms for e in estimates])

    def stat(fn, values):
        return float(fn(values)) if len(values) else float("nan")

    return {
        "frames": n,
        "availability": float(trk.mean()) if n else 0.0,
        "time_to_fix_s": float(truth[first[0]].t - truth[0].t) if len(first) else float("nan"),
        "err_median_m": stat(np.median, err[trk]),
        "err_p95_m": stat(lambda v: np.percentile(v, 95), err[trk]),
        "heading_err_median_deg": stat(np.median, herr[trk]),
        "recall_10m": float(np.mean(trk & (err < 10))) if n else 0.0,
        "recall_25m": float(np.mean(trk & (err < 25))) if n else 0.0,
        "confident_wrong": float(np.mean(err[trk] > WRONG_M)) if trk.any() else 0.0,
        "latency_median_ms": stat(np.median, latency),
    }
