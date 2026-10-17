"""Offline localization of recorded footage, with exports and a GNSS comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from pharos.eval.export import record, to_csv, to_geojson, to_kml
from pharos.eval.metrics import WRONG_M, integrity
from pharos.eval.plot import draw_track
from pharos.footage import Footage
from pharos.geomap import GeoMap
from pharos.nav.localizer import Localizer, LocalizerConfig
from pharos.types import Estimate, GroundTruth, Status


def analyze(
    footage: Footage,
    geomap: GeoMap,
    cfg: LocalizerConfig | None = None,
    progress=None,
    smooth: bool = True,
) -> list[dict]:
    """One record per frame (see :func:`pharos.eval.export.record`).

    Recorded footage need not be localized causally: with ``smooth`` the positions
    come from one solve over the whole flight, so gaps are bridged from both sides.
    Each record then keeps what a live run would have reported under ``causal``.
    """
    localizer = Localizer(geomap, footage.recording.camera, cfg)
    records, telemetry = [], []
    for i, (frame, gnss) in enumerate(zip(footage.recording, footage.gnss, strict=True)):
        records.append(record(localizer.process(frame), frame.telemetry, gnss, geomap))
        telemetry.append(frame.telemetry)
        if progress:
            progress(i + 1, len(footage.gnss), records[-1])
    if not smooth:
        return records
    out = []
    for causal, e, tel, gnss in zip(
        records, localizer.smooth(), telemetry, footage.gnss, strict=True
    ):
        rec = record(e, tel, gnss, geomap)
        keep = ("status", "x", "y", "sigma_m", "error_m")
        rec["causal"] = {k: causal[k] for k in keep if k in causal}
        out.append(rec)
    return out


def summarize(records: list[dict]) -> dict:
    """Coverage, and accuracy against the aircraft's own GNSS where it reported one.

    For smoothed records, the causal (live) result is summarized under ``causal``.
    """
    if records and "causal" in records[0]:
        causal = [r["causal"] | {k: r[k] for k in ("t", "gnss") if k in r} for r in records]
        return _summarize(records) | {"causal": _summarize(causal)}
    return _summarize(records)


def _summarize(records: list[dict]) -> dict:
    n = len(records)
    trusted = [r for r in records if r["status"] == Status.TRACKING.value]
    errors = np.array([r["error_m"] for r in trusted if "error_m" in r])
    out: dict = {
        "frames": n,
        "tracking": len(trusted) / n if n else 0.0,
        "first_fix_s": trusted[0]["t"] - records[0]["t"] if trusted else None,
        "with_gnss": sum("gnss" in r for r in records),
    }
    if len(errors):
        out |= {
            "error_median_m": float(np.median(errors)),
            "error_p95_m": float(np.percentile(errors, 95)),
            "confident_wrong": float(np.mean(errors > WRONG_M)),
        }
    scored = [r for r in records if "gnss" in r]  # lost frames count, as uncovered
    if scored:
        trusted = np.array([r["status"] == Status.TRACKING.value for r in scored])
        sigma = np.array([r.get("sigma_m", np.inf) for r in scored])
        err = np.array([r.get("error_m", np.inf) for r in scored])
        located = np.array([r["status"] != Status.LOST.value and "x" in r for r in scored])
        out |= integrity(np.array([r["t"] for r in scored]), err, trusted, sigma, located)
    return out


def write_outputs(records: list[dict], geomap: GeoMap, out: Path, name: str) -> dict:
    """Track files, an overview image and a summary in ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "track.geojson").write_text(to_geojson(records, name))
    (out / "track.kml").write_text(to_kml(records, name))
    (out / "track.csv").write_text(to_csv(records))
    with open(out / "frames.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    estimates = [
        Estimate(r["t"], Status(r["status"]), r.get("x", math.nan), r.get("y", math.nan))
        for r in records
    ]
    gnss = [r for r in records if "gnss" in r]
    truth = [GroundTruth(r["t"], r["gnss"]["x"], r["gnss"]["y"], 0, 0) for r in gnss]
    cv2.imwrite(str(out / "overview.png"), draw_track(geomap, estimates, truth or None))
    summary = summarize(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
