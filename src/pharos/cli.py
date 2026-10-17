"""Command line: ``pharos analyze | probe | serve | simulate | localize | bench``."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2

from pharos.analyze import analyze, write_outputs
from pharos.config import load_yaml
from pharos.eval.bench import localize, run_scenarios, table_header, table_row
from pharos.eval.metrics import evaluate
from pharos.eval.plot import draw_track
from pharos.eval.probe import probe, summarize_probe
from pharos.footage import load_footage
from pharos.geomap import GeoMap
from pharos.nav.localizer import LocalizerConfig
from pharos.recording import load_recording, save_recording
from pharos.sim.flight import simulate
from pharos.sim.scenario import builtin_scenarios, load_scenario
from pharos.telemetry import DataflashTelemetry, SrtTelemetry
from pharos.types import Estimate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pharos", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="render a scenario to a map and a flight recording")
    p.add_argument("scenario", help="built-in scenario name or YAML path")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.set_defaults(func=_simulate)

    p = sub.add_parser("localize", help="localize a recording against a map")
    p.add_argument("--map", type=Path, required=True, help="map .json sidecar")
    p.add_argument("--recording", type=Path, required=True)
    p.add_argument("-c", "--config", type=Path, help="localizer YAML overrides")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.set_defaults(func=_localize)

    p = sub.add_parser("bench", help="simulate and localize scenarios, print a results table")
    p.add_argument("scenarios", nargs="*", help="names or YAML paths (default: all built-in)")
    p.add_argument("-c", "--config", type=Path, help="localizer YAML overrides")
    p.add_argument("-o", "--out", type=Path, help="write metrics and track images here")
    p.add_argument("-j", "--jobs", type=int, default=1, help="scenarios to run in parallel")
    p.set_defaults(func=_bench)

    p = sub.add_parser("analyze", help="localize recorded footage and export the track")
    _footage_args(p)
    p.add_argument("-c", "--config", type=Path, help="localizer YAML overrides")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.set_defaults(func=_analyze)

    p = sub.add_parser(
        "probe", help="per-frame match quality on footage with GNSS (for tuning on real data)"
    )
    _footage_args(p)
    p.add_argument("-c", "--config", type=Path, help="localizer YAML overrides")
    p.add_argument("--every", type=int, default=1, help="use every Nth frame")
    p.add_argument("-o", "--out", type=Path, help="write per-frame rows and summary as JSON")
    p.set_defaults(func=_probe)

    p = sub.add_parser("serve", help="live service: ingest feeds, dashboard at http://HOST:PORT")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept other devices")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data", type=Path, default=Path("pharos-data"), help="maps and uploads")
    p.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    p.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    args.func(args)


def _footage_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("footage", type=Path, help="video file or folder of photos")
    p.add_argument("--map", type=Path, required=True, help="GeoTIFF or map .json")
    p.add_argument("--hfov", type=float, help="horizontal field of view, degrees")
    p.add_argument("--altitude", type=float, help="height above ground, if the footage has none")
    p.add_argument(
        "--ground", type=float, help="ground height in the photos' GPS altitude reference"
    )
    p.add_argument("--srt", type=Path, help="DJI subtitle file (default: next to the video)")
    p.add_argument("--log", type=Path, help="ArduPilot DataFlash log (.bin) of the flight")
    p.add_argument(
        "--log-sync",
        type=float,
        default=0.0,
        help="seconds from the log's first GNSS fix to the start of the video",
    )
    p.add_argument("--start", type=float, default=0.0, help="video: skip to this time, seconds")
    p.add_argument("--end", type=float, help="video: stop at this time, seconds")
    p.add_argument(
        "--mount-offset",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("FORWARD", "RIGHT"),
        help="camera tilt the telemetry misses, degrees (pharos probe measures it)",
    )
    p.add_argument("--rate", type=float, default=2.0, help="video frames analyzed per second")
    p.add_argument("--map-gsd", type=float, default=1.0, help="matching resolution, m/px")


def _load_footage_and_map(args):
    telemetry = None
    if args.srt:
        telemetry = SrtTelemetry(args.srt)
    elif args.log:
        telemetry = DataflashTelemetry(args.log, sync_s=args.log_sync)
    footage = load_footage(
        args.footage,
        hfov_deg=args.hfov,
        altitude_m=args.altitude,
        ground_m=args.ground,
        telemetry=telemetry,
        rate_hz=args.rate,
        start_s=args.start,
        end_s=args.end,
        mount_offset_deg=tuple(args.mount_offset),
    )
    return footage, GeoMap.open(args.map).resampled(args.map_gsd)


def _analyze(args) -> None:
    footage, geomap = _load_footage_and_map(args)

    def progress(i: int, n: int, rec: dict) -> None:
        err = f"  error vs GNSS {rec['error_m']:.0f} m" if "error_m" in rec else ""
        print(f"\r{i}/{n} {rec['status']:<10}{err}      ", end="", flush=True)

    records = analyze(footage, geomap, _config(args.config), progress)
    print()
    summary = write_outputs(records, geomap, args.out, args.footage.name)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}: track.geojson, track.kml, track.csv, overview.png")


def _probe(args) -> None:
    footage, geomap = _load_footage_and_map(args)

    def progress(i: int, n: int, row: dict) -> None:
        print(f"\r{i}/{n}", end="", flush=True)

    rows = probe(footage, geomap, _config(args.config), every=args.every, progress=progress)
    print()
    summary = summarize_probe(rows)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "frames": rows}, indent=1))


def _simulate(args) -> None:
    geomap, rec = simulate(load_scenario(args.scenario))
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
    scenarios = [load_scenario(s) for s in args.scenarios or builtin_scenarios()]
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


def _serve(args) -> None:
    import threading
    import webbrowser

    import uvicorn

    from pharos.server.app import create_app

    if args.open:
        host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
        url = f"http://{host}:{args.port}"
        threading.Timer(1.5, webbrowser.open, [url]).start()
    uvicorn.run(create_app(args.data), host=args.host, port=args.port)


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
