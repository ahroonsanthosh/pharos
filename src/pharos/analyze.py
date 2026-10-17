"""Offline localization of recorded footage, with exports and a GNSS comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from pharos.eval.export import record, to_csv, to_geojson, to_kml
from pharos.eval.metrics import WRONG_M
from pharos.eval.plot import draw_track
from pharos.footage import Footage
from pharos.geomap import GeoMap
from pharos.nav.localizer import Localizer, LocalizerConfig
from pharos.types import Estimate, GroundTruth, Status


def analyze(
    footage: Footage, geomap: GeoMap, cfg: LocalizerConfig | None = None, progress=None
) -> list[dict]:
    """One record per frame (see :func:`pharos.eval.export.record`)."""
    localizer = Localizer(geomap, footage.recording.camera, cfg)
    records = []
    for i, (frame, gnss) in enumerate(zip(footage.recording, footage.gnss, strict=True)):
        records.append(record(localizer.process(frame), frame.telemetry, gnss, geomap))
        if progress:
            progress(i + 1, len(footage.gnss), records[-1])
    return records


def summarize(records: list[dict]) -> dict:
    """Coverage, and accuracy against the aircraft's own GNSS where it reported one."""
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
