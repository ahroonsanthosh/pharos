"""Accuracy, availability and integrity metrics for a localized flight.

The headline pair is accurate coverage and the misleading rate, both over all
frames: how often the localizer vouches for a position that is within
``WRONG_M`` of the truth, and how often it vouches for one that is not. A
trusted position must also carry an honest uncertainty, so the share of
trusted frames within three reported sigmas is reported too, and the longest
stretch without any trusted position, which is what "continuously tracked"
asks for.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pharos.geometry import wrap_angle
from pharos.types import Estimate, GroundTruth, Status

WRONG_M = 25.0  # a TRACKING estimate further off than this counts as confidently wrong


def integrity(
    t: np.ndarray,
    err: np.ndarray,
    trusted: np.ndarray,
    sigma: np.ndarray | None = None,
    located: np.ndarray | None = None,
) -> dict[str, float]:
    """Coverage, misleading rate, sigma calibration and the longest untrusted gap.

    ``located`` marks frames with any position (trusted or coasting): with an honest
    sigma, a coasting position is still a usable one.
    """
    n = len(t)
    out = {
        "coverage_25m": float(np.mean(trusted & (err <= WRONG_M))) if n else 0.0,
        "misleading": float(np.mean(trusted & (err > WRONG_M))) if n else 0.0,
        "longest_gap_s": _longest_gap(t, trusted),
    }
    if sigma is not None and trusted.any():
        out["within_3_sigma"] = float(np.mean(err[trusted] <= 3 * sigma[trusted]))
    if located is not None and n:
        out["located_25m"] = float(np.mean(located & (err <= WRONG_M)))
        if sigma is not None and located.any():
            out["located_within_3_sigma"] = float(np.mean(err[located] <= 3 * sigma[located]))
    return out


def _longest_gap(t: np.ndarray, trusted: np.ndarray) -> float:
    """Longest time between trusted frames (or from the start / to the end)."""
    if not len(t):
        return 0.0
    marks = np.concatenate([[t[0]], t[trusted], [t[-1]]])
    return float(np.max(np.diff(marks))) if len(marks) > 1 else 0.0


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
    } | integrity(
        np.array([g.t for g in truth]),
        err,
        trk,
        np.array([e.sigma_xy for e in estimates]),
        np.array([e.status is not Status.LOST and np.isfinite(e.x) for e in estimates]),
    )
