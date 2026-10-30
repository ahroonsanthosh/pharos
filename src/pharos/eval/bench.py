"""Run scenarios end to end and tabulate the results."""

from __future__ import annotations

import math
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat

import numpy as np

from pharos.eval.metrics import evaluate
from pharos.eval.plot import draw_track
from pharos.geomap import GeoMap
from pharos.nav.localizer import Localizer, LocalizerConfig
from pharos.recording import Recording
from pharos.sim.flight import simulate
from pharos.sim.scenario import Scenario
from pharos.types import Estimate


@dataclass
class RunResult:
    name: str
    metrics: dict[str, float]
    track: np.ndarray  # estimated vs true track drawn over the map


def localize(
    geomap: GeoMap,
    rec: Recording,
    cfg: LocalizerConfig | None = None,
    start: tuple | None = None,
) -> list[Estimate]:
    """Estimates for every frame; ``start`` is a known first position, its sigma and
    optionally the velocity (east, north) the aircraft had there."""
    loc = Localizer(geomap, rec.camera, cfg, takeoff=rec.takeoff)
    if start is not None:
        xy, sigma, *velocity = start
        loc.start_at(xy, sigma, velocity=velocity[0] if velocity else None)
    return [loc.process(frame) for frame in rec]


def run_scenario(sc: Scenario, cfg: LocalizerConfig | None = None) -> RunResult:
    geomap, rec = simulate(sc)
    start = None
    if sc.known_start_m is not None and len(rec.truth) > 1:
        off = np.random.default_rng(sc.seed).normal(0.0, sc.known_start_m, 2)
        if sc.known_start_error_m is not None:
            off = np.asarray(sc.known_start_error_m, float)
        a, b = rec.truth[:2]
        velocity = ((b.x - a.x) / (b.t - a.t), (b.y - a.y) / (b.t - a.t))
        start = ((a.x + off[0], a.y + off[1]), sc.known_start_m, velocity)
    estimates = localize(geomap, rec, cfg, start)
    assert rec.truth is not None
    track = draw_track(geomap, estimates, rec.truth)
    return RunResult(sc.name, evaluate(estimates, rec.truth), track)


def run_scenarios(
    scenarios: list[Scenario], cfg: LocalizerConfig | None = None, jobs: int = 1
) -> Iterator[RunResult]:
    """Results in input order, computed in ``jobs`` worker processes."""
    if jobs <= 1:
        yield from (run_scenario(sc, cfg) for sc in scenarios)
        return
    with ProcessPoolExecutor(jobs) as pool:
        yield from pool.map(run_scenario, scenarios, repeat(cfg))


_COLUMNS = [
    ("availability", "avail", "{:.0%}"),
    ("time_to_fix_s", "fix s", "{:.1f}"),
    ("err_median_m", "err50 m", "{:.1f}"),
    ("err_p95_m", "err95 m", "{:.1f}"),
    ("heading_err_median_deg", "hdg deg", "{:.1f}"),
    ("coverage_25m", "cover", "{:.0%}"),
    ("misleading", "mislead", "{:.1%}"),
    ("longest_gap_s", "gap s", "{:.0f}"),
    ("within_3_sigma", "in 3 sd", "{:.0%}"),
    ("latency_median_ms", "ms/frame", "{:.0f}"),
]


def table_header() -> str:
    head = "| scenario | " + " | ".join(c[1] for c in _COLUMNS) + " |"
    return head + "\n|---|" + "---:|" * len(_COLUMNS)


def table_row(r: RunResult) -> str:
    # a flight never trusted has no trusted positions to check against their sigmas
    values = [(r.metrics.get(key, math.nan), fmt) for key, _, fmt in _COLUMNS]
    cells = ["-" if math.isnan(v) else fmt.format(v) for v, fmt in values]
    return f"| {r.name} | " + " | ".join(cells) + " |"
