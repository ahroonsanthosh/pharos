"""Command line: ``pharos simulate | localize | bench``."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2

from pharos.config import load_yaml
from pharos.eval.bench import localize, run_scenarios, table_header, table_row
from pharos.eval.metrics import evaluate
from pharos.eval.plot import draw_track
from pharos.geomap import GeoMap
from pharos.nav.localizer import LocalizerConfig
from pharos.recording import load_recording, save_recording
from pharos.sim.flight import simulate
from pharos.sim.scenario import Scenario
from pharos.types import Estimate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pharos", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="render a scenario to a map and a flight recording")
    p.add_argument("scenario", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.set_defaults(func=_simulate)

    p = sub.add_parser("localize", help="localize a recording against a map")
    p.add_argument("--map", type=Path, required=True, help="map .json sidecar")
    p.add_argument("--recording", type=Path, required=True)
    p.add_argument("-c", "--config", type=Path, help="localizer YAML overrides")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.set_defaults(func=_localize)

    p = sub.add_parser("bench", help="simulate and localize scenarios, print a results table")
    p.add_argument("scenarios", type=Path, nargs="+")
    p.add_argument("-c", "--config", type=Path, help="localizer YAML overrides")
    p.add_argument("-o", "--out", type=Path, help="write metrics and track images here")
    p.add_argument("-j", "--jobs", type=int, default=1, help="scenarios to run in parallel")
    p.set_defaults(func=_bench)

    args = parser.parse_args(argv)
    args.func(args)


def _simulate(args) -> None:
    geomap, rec = simulate(load_yaml(Scenario, args.scenario))
    geomap.save(args.out / "map")
    save_recording(rec, args.out / "recording")
    print(f"wrote {args.out / 'map.json'} and {len(rec)} frames to {args.out / 'recording'}")


def _localize(args) -> None:
    geomap = GeoMap.load(args.map)
    rec = load_recording(args.recording)
    estimates = localize(geomap, rec, _config(args.config))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "estimates.jsonl", "w") as f:
        for e in estimates:
            f.write(json.dumps(_estimate_json(e, geomap)) + "\n")
    cv2.imwrite(str(args.out / "track.png"), draw_track(geomap, estimates, rec.truth))
    if rec.truth is not None:
        metrics = evaluate(estimates, rec.truth)
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out}")


def _bench(args) -> None:
    scenarios = [load_yaml(Scenario, path) for path in args.scenarios]
    lines = [table_header()]
    print(lines[0], flush=True)
    metrics = {}
    for result in run_scenarios(scenarios, _config(args.config), args.jobs):
        lines.append(table_row(result))
        metrics[result.name] = result.metrics
        print(lines[-1], flush=True)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.out / f"{result.name}.png"), result.track)
    if args.out:
        (args.out / "results.md").write_text("\n".join(lines) + "\n")
        (args.out / "results.json").write_text(json.dumps(metrics, indent=2))


def _config(path: Path | None) -> LocalizerConfig:
    return load_yaml(LocalizerConfig, path) if path else LocalizerConfig()


def _estimate_json(e: Estimate, geomap: GeoMap) -> dict:
    out = {"t": e.t, "status": e.status.value}
    if e.status.value != "lost":
        out |= {
            "x": e.x,
            "y": e.y,
            "heading_deg": math.degrees(e.heading) % 360,
            "sigma_m": e.sigma_xy,
        }
        latlon = geomap.to_latlon(e.x, e.y)
        if latlon:
            out |= {"lat": latlon[0], "lon": latlon[1]}
    if e.fix is not None:
        out["score"] = e.fix.score
    out["latency_ms"] = e.latency_ms
    return out


if __name__ == "__main__":
    main()
