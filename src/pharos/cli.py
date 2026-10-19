"""Command line: ``pharos analyze | probe | sync | serve | simulate | localize | bench``."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
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
from pharos.sync import estimate_sync
from pharos.telemetry import DataflashTelemetry, SrtTelemetry
from pharos.terrain import Terrain
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
    p.add_argument(
        "--causal", action="store_true", help="only what a live run would report (no smoothing)"
    )
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

    p = sub.add_parser("sync", help="find where a video starts in its ArduPilot log")
    p.add_argument("video", type=Path)
    p.add_argument("--log", type=Path, required=True, help="ArduPilot DataFlash log (.bin)")
    p.add_argument("--log-sync", type=float, default=0.0, help="first guess, seconds")
    p.add_argument("--hfov", type=float, required=True, help="horizontal field of view, degrees")
    p.add_argument("--start", type=float, default=0.0, help="video time to start at, seconds")
    p.add_argument("--end", type=float, help="video time to stop at (default: start + 120 s)")
    p.add_argument("--span", type=float, default=2.0, help="search this far around the guess")
    p.set_defaults(func=_sync)

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
    p.add_argument(
        "--log-height",
        choices=("auto", "home", "terrain"),
        default="auto",
        help="ArduPilot heights: barometric above home, corrected with the terrain model "
        "(home); the autopilot's terrain database, looked up at its GNSS position (terrain); "
        "home if there is a terrain model, else terrain (auto)",
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
    p.add_argument(
        "--takeoff",
        type=float,
        nargs=2,
        metavar=("LAT", "LON"),
        help="where the aircraft took off (default: from the telemetry, if it starts there)",
    )
    p.add_argument(
        "--dem",
        type=Path,
        help="terrain model GeoTIFF (default: Copernicus GLO-30, downloaded once and cached)",
    )
    p.add_argument(
        "--no-terrain",
        action="store_true",
        help="take heights above takeoff as heights above ground",
    )
    p.add_argument("--rate", type=float, default=2.0, help="video frames analyzed per second")
    p.add_argument("--map-gsd", type=float, default=1.0, help="matching resolution, m/px")
    p.add_argument(
        "--map-accuracy",
        type=float,
        default=0.0,
        help="the map's own position error, m (e.g. 6 for NAIP), added to reported sigmas",
    )


def _load_footage_and_map(args):
    telemetry = None
    if args.srt:
        telemetry = SrtTelemetry(args.srt)
    elif args.log:
        terrain_database = args.log_height == "terrain"
        telemetry = DataflashTelemetry(args.log, args.log_sync, terrain_database=terrain_database)

    def load():
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
        if args.takeoff:
            footage.takeoff = tuple(args.takeoff)
        return footage

    footage = load()
    geomap = GeoMap.open(args.map).resampled(args.map_gsd)
    if args.no_terrain:
        geomap.terrain = None
    elif any(t.above_takeoff for t in footage.telemetry) and (args.dem or not geomap.terrain):
        geomap.terrain = _terrain(geomap, args.dem, footage.takeoff)
    if args.log and args.log_height == "auto" and geomap.terrain is None:
        telemetry.terrain_database = True  # no terrain model: the autopilot's, if it logged one
        footage = load()
    return footage, geomap


def _terrain(geomap: GeoMap, dem: Path | None, takeoff) -> Terrain | None:
    """The terrain under the map (and the takeoff point), or None if there is none to be had.

    A map saved with its terrain (``GeoMap.save``) keeps it unless ``--dem`` is given.
    """
    include = []
    if takeoff is not None and (xy := geomap.from_latlon(*takeoff)) is not None:
        include.append(xy)
    try:
        if dem is not None:
            return Terrain.from_dem([dem], geomap, include)
        print("terrain: Copernicus GLO-30 (the first use downloads about 40 MB per degree tile)")
        return Terrain.copernicus(geomap, include)
    except (ImportError, OSError, ValueError) as exc:
        print(f"warning: no terrain ({exc}); heights above takeoff taken as above ground")
        return None


def _analyze(args) -> None:
    footage, geomap = _load_footage_and_map(args)

    def progress(i: int, n: int, rec: dict) -> None:
        err = f"  error vs GNSS {rec['error_m']:.0f} m" if "error_m" in rec else ""
        print(f"\r{i}/{n} {rec['status']:<10}{err}      ", end="", flush=True)

    cfg = _with_map_accuracy(_config(args.config), args.map_accuracy)
    records = analyze(footage, geomap, cfg, progress, smooth=not args.causal)
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


def _sync(args) -> None:
    telemetry = DataflashTelemetry(args.log, sync_s=args.log_sync)
    end = args.end if args.end is not None else args.start + 120.0
    footage = load_footage(
        args.video, hfov_deg=args.hfov, telemetry=telemetry, start_s=args.start, end_s=end
    )
    result = estimate_sync(footage.recording, telemetry, footage.recording.camera, args.span)
    for offset, jitter in result.curve:
        print(f"  {args.log_sync + offset:7.2f} s  odometry jitter {jitter:6.2f} m")
    print(
        f"--log-sync {args.log_sync + result.offset_s:.2f}  (odometry jitter "
        f"{result.jitter_m:.2f} m, {result.initial_jitter_m:.2f} m at {args.log_sync:.2f})"
    )
    if result.contrast < 0.15:
        print("warning: weak minimum; pick a stretch with turns, or a longer one")


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


def _with_map_accuracy(cfg: LocalizerConfig, accuracy_m: float) -> LocalizerConfig:
    """A map that is itself off by metres makes every match off alike: systematic error."""
    systematic = math.hypot(cfg.window.systematic_sigma_m, accuracy_m)
    return replace(cfg, window=replace(cfg.window, systematic_sigma_m=systematic))


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
