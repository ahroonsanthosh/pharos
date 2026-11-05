"""HTTP API and dashboard for live localization (``pharos serve``)."""

from __future__ import annotations

import itertools
import math
import os
import shutil
import socket
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pharos import __version__
from pharos.config import from_dict
from pharos.eval.export import to_csv, to_geojson, to_kml
from pharos.server.feeds import Feed, FeedSpec, check_spec
from pharos.server.store import MapStore, safe_name
from pharos.sim.scenario import SCENARIO_DIR, builtin_scenarios
from pharos.telemetry import ManualTelemetry

STATIC = Path(__file__).parent / "static"
# Public mode (PHAROS_PUBLIC=1, for a server anyone can reach): visitors may only run
# the built-in demo flights, their own browser camera, or footage they upload here.
# No paths on the server, stream URLs, attached cameras or MAVLink, and bounded use.
PUBLIC_MAX_UPLOAD_MB = 500  # per file
PUBLIC_MAX_UPLOADS_MB = 4000  # everything uploaded, maps included
PUBLIC_MAX_ACTIVE = 2  # feeds running at once
PUBLIC_MAX_FEEDS = 12  # feeds kept listed; the oldest finished ones go first
PUBLIC_MAX_FEED_S = 15 * 60  # a feed is stopped after this long
PUBLIC_MAX_RATE_HZ = 4.0

# Demo flights as the dashboard lists them: the everyday ones first, in this order,
# then the stress tests that put one thing that goes wrong in a real flight at a time.
DEMO_FLIGHTS = {
    "nominal": "Summer farmland",
    "valley": "Steep valleys, tilted camera",
    "hills": "Hill survey following the terrain",
    "ireland_pasture": "Irish pasture on a grey day",
    "ireland_bog_coast": "Irish bog and coast",
    "ireland_uplands": "Irish uplands",
    "winter_snow": "Snow cover",
    "autumn_low_sun": "Autumn, low sun",
    "dusk_haze": "Dusk haze",
    "grid_farmland": "Grid farmland",
    "land_change": "Years since the map was made",
    "spring_clouds_on_map": "Clouds on the map",
    "tilted_camera": "Forward-tilted camera",
}
STRESS_TESTS = {
    "stress_gnss_lost_low": "GPS jammed at 50 m",
    "stress_spoofed_start": "Spoofed last GPS fix, 60 m off",
    "stress_spoofed_start_far": "Spoofed last GPS fix, 120 m off",
    "stress_midflight": "Joined in mid-air",
    "stress_video": "Frozen video link",
    "stress_telemetry": "Telemetry dropout",
    "stress_rain": "Rain on the lens",
    "stress_hover": "Hovering in place",
    "stress_descent": "Descending to 45 m",
    "stress_low_sun_low": "Low winter sun at 50 m",
    "stress_off_map": "Flying off the map",
    "stress_map_offset": "Map 10 m off",
    "stress_tide": "A different tide",
    "stress_takeoff_hill": "Took off from a hill",
    "altitude_bias": "Barometer 12% off",
    "sensor_errors": "Telemetry errors",
    "no_compass_spinning": "No compass, spinning",
    "kitchen_sink": "Everything at once",
}

_EXPORTS = {
    "geojson": (to_geojson, "application/geo+json"),
    "kml": (to_kml, "application/vnd.google-earth.kml+xml"),
    "csv": (to_csv, "text/csv"),
}


def create_app(data_dir: str | Path, public: bool | None = None) -> FastAPI:
    if public is None:
        public = os.environ.get("PHAROS_PUBLIC", "") not in ("", "0", "false")
    data = Path(data_dir)
    uploads = data / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    store = MapStore(data / "maps")
    feeds: dict[str, Feed] = {}
    started: dict[str, float] = {}  # feed id -> monotonic start time
    ids = itertools.count(1)
    limit = PUBLIC_MAX_UPLOAD_MB * 2**20 if public else None

    def check_space() -> None:
        if public and _size(data) > PUBLIC_MAX_UPLOADS_MB * 2**20:
            raise HTTPException(507, "this public server is full; try again later")

    def tidy() -> None:
        """Public mode: stop feeds that ran too long, forget the oldest finished ones."""
        now = time.monotonic()
        for fid, feed in feeds.items():
            if feed.state in ("starting", "running") and now - started[fid] > PUBLIC_MAX_FEED_S:
                feed.stop()
        done = [fid for fid, feed in feeds.items() if feed.state not in ("starting", "running")]
        while len(feeds) > PUBLIC_MAX_FEEDS and done:
            fid = done.pop(0)
            del feeds[fid], started[fid]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        for feed in feeds.values():
            feed.stop()

    app = FastAPI(title="Pharos", version=__version__, docs_url="/api/docs", lifespan=lifespan)
    app.state.store = store
    app.state.feeds = feeds
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def get_feed(feed_id: str) -> Feed:
        if feed_id not in feeds:
            raise HTTPException(404, f"no feed {feed_id}")
        return feeds[feed_id]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info() -> dict:
        if public:
            return {"public": True, "lan_ips": []}
        return {"public": False, "lan_ips": _lan_ips(), "data_dir": str(data.resolve())}

    # -- maps -------------------------------------------------------------

    @app.get("/api/maps")
    def list_maps() -> list[dict]:
        return [store.info(name) for name in store.names()]

    @app.post("/api/maps")
    def upload_map(name: str, files: list[UploadFile]) -> dict:
        """A GeoTIFF, or a map ``.json`` sidecar together with its image."""
        check_space()
        with tempfile.TemporaryDirectory() as tmp:
            saved = [_save_upload(f, Path(tmp), limit) for f in files]
            main = next((p for p in saved if p.suffix.lower() in (".tif", ".tiff", ".json")), None)
            if main is None:
                raise HTTPException(422, "upload a GeoTIFF or a .json sidecar with its image")
            try:
                stored = store.add_file(name, main)
            except (ValueError, OSError, KeyError) as exc:
                reason = str(exc).replace(f"{tmp}/", "").replace(f"{tmp}\\", "")
                raise HTTPException(422, f"could not read {main.name} as a map: {reason}") from exc
        return store.info(stored)

    @app.get("/api/maps/{name}/preview.jpg")
    def map_preview(name: str) -> Response:
        try:
            return Response(store.preview(name), media_type="image/jpeg")
        except KeyError as exc:
            raise HTTPException(404, exc.args[0]) from exc

    @app.get("/api/scenarios")
    def scenarios() -> list[dict]:
        """Built-in demo flights: name, a readable title, its group and what it shows."""
        return [_demo_info(name) for name in _demo_order()]

    @app.post("/api/uploads")
    def upload_files(files: list[UploadFile]) -> dict:
        """Store a video, an SRT file, or a set of photos (kept together in a new folder).

        Returns the server path to use as a feed source or telemetry file.
        """
        check_space()
        if len(files) == 1:
            return {"path": str(_save_upload(files[0], uploads, limit))}
        folder = Path(tempfile.mkdtemp(prefix="photos-", dir=uploads))
        for f in files:
            _save_upload(f, folder, limit)
        return {"path": str(folder)}

    @app.get("/api/feeds/{feed_id}/export.{fmt}")
    def export_track(feed_id: str, fmt: str) -> Response:
        """Download the feed's track as GeoJSON, KML (Google Earth) or CSV."""
        feed = get_feed(feed_id)
        name = safe_name(feed.spec.name or feed.id)
        if fmt not in _EXPORTS:
            raise HTTPException(404, f"unknown format {fmt}; use geojson, kml or csv")
        render, media_type = _EXPORTS[fmt]
        body = render(list(feed.records)) if fmt == "csv" else render(list(feed.records), name)
        headers = {"content-disposition": f'attachment; filename="{name}.{fmt}"'}
        return Response(body, media_type=media_type, headers=headers)

    # -- feeds ------------------------------------------------------------

    @app.get("/api/feeds")
    def list_feeds() -> list[dict]:
        if public:
            tidy()
        return [f.summary() for f in feeds.values()]

    @app.post("/api/feeds")
    def create_feed(body: dict) -> dict:
        try:
            spec = from_dict(FeedSpec, body)
            if public:
                _check_public(spec, uploads)  # what is not allowed here at all comes first
            check_spec(spec, store.names(), builtin_scenarios())
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if public:
            tidy()
            if sum(f.state in ("starting", "running") for f in feeds.values()) >= PUBLIC_MAX_ACTIVE:
                raise HTTPException(429, "this public server is busy; stop a feed or try later")
        feed_id = f"feed-{next(ids)}"
        feeds[feed_id] = Feed(feed_id, spec, store)
        started[feed_id] = time.monotonic()
        return feeds[feed_id].summary()

    @app.delete("/api/feeds")
    def clear_feeds(running: bool = False) -> dict:
        """Remove finished feeds; with ``running``, stop and remove the running ones too.

        A public server only clears finished feeds: running ones may be someone else's.
        """
        cleared = []
        for fid, feed in list(feeds.items()):
            active = feed.state in ("starting", "running")
            if active and not (running and not public):
                continue
            if active:
                feed.stop()
            del feeds[fid]
            started.pop(fid, None)
            cleared.append(fid)
        return {"cleared": cleared}

    @app.delete("/api/feeds/{feed_id}")
    def delete_feed(feed_id: str) -> dict:
        feed = get_feed(feed_id)
        feed.stop()
        del feeds[feed_id]
        started.pop(feed_id, None)
        return {"deleted": feed_id}

    @app.get("/api/feeds/{feed_id}/records")
    def feed_records(feed_id: str, since: int = 0) -> dict:
        feed = get_feed(feed_id)
        return {**feed.summary(), "since": since, "records": feed.records[since:]}

    @app.get("/api/feeds/{feed_id}/preview.jpg")
    def feed_preview(feed_id: str) -> Response:
        feed = get_feed(feed_id)
        if feed.preview is None:
            raise HTTPException(404, "no frame yet")
        return Response(feed.preview, media_type="image/jpeg")

    @app.patch("/api/feeds/{feed_id}/telemetry")
    def update_telemetry(feed_id: str, body: dict) -> dict:
        """Change manual altitude / tilt while the feed runs."""
        feed = get_feed(feed_id)
        if not isinstance(feed.telemetry, ManualTelemetry):
            raise HTTPException(409, "feed does not use manual telemetry")
        try:
            now = feed.telemetry.telemetry
            altitude = float(body.get("altitude_m", now.altitude))
            tilt = float(body.get("tilt_deg", math.degrees(now.pitch)))
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, "height and tilt must be numbers") from exc
        if not (math.isfinite(altitude) and altitude > 0):
            raise HTTPException(422, "height above ground must be more than 0 m")
        if not -90 <= tilt <= 90:
            raise HTTPException(422, "camera tilt must be between -90 and 90 degrees")
        feed.telemetry.update(altitude=altitude, pitch=math.radians(tilt))
        return {"ok": True}

    @app.websocket("/api/feeds/{feed_id}/frames")
    async def push_frames(ws: WebSocket, feed_id: str) -> None:
        """Receive JPEG frames from the dashboard (browser camera or capture card)."""
        await ws.accept()
        feed = feeds.get(feed_id)
        if feed is None or feed.pushed is None:
            await ws.close(code=4404)
            return
        try:
            while True:
                feed.pushed.put(await ws.receive_bytes())
        except WebSocketDisconnect:
            pass

    return app


def _save_upload(upload: UploadFile, folder: Path, limit: int | None = None) -> Path:
    path = folder / safe_name(Path(upload.filename or "upload").name)
    with open(path, "wb") as out:
        if limit is None:
            shutil.copyfileobj(upload.file, out)
            return path
        written = 0
        while chunk := upload.file.read(2**20):
            written += len(chunk)
            if written > limit:
                out.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, f"files over {limit // 2**20} MB are not accepted here")
            out.write(chunk)
    return path


def _demo_order() -> list[str]:
    known = [*DEMO_FLIGHTS, *STRESS_TESTS]
    present = set(builtin_scenarios())
    return [n for n in known if n in present] + sorted(present - set(known))


def _demo_info(name: str) -> dict:
    if name in DEMO_FLIGHTS:
        title, group = DEMO_FLIGHTS[name], "Demo flights"
    else:
        title, group = STRESS_TESTS.get(name, name.replace("_", " ").capitalize()), "Stress tests"
    return {"name": name, "title": title, "group": group, "about": _about(name)}


def _about(name: str) -> str:
    """The scenario file's opening comment: what the flight shows."""
    lines = []
    for line in (SCENARIO_DIR / f"{name}.yaml").read_text().splitlines():
        if not line.startswith("#"):
            break
        lines.append(line.lstrip("# ").strip())
    return " ".join(lines)


def _check_public(spec: FeedSpec, uploads: Path) -> None:
    """Refuse what a public server must not do for a visitor."""

    def uploaded(path: str) -> bool:
        p = Path(path).resolve()
        return p.is_relative_to(uploads.resolve()) and p.exists()

    source = spec.source
    if source.startswith("demo:"):
        if source.removeprefix("demo:") not in builtin_scenarios():
            raise HTTPException(403, "unknown demo flight")
    elif source != "browser" and not uploaded(source):
        raise HTTPException(
            403, "this public server runs demo flights, your camera and uploads only"
        )
    kind, _, arg = spec.telemetry.partition(":")
    if not (kind == "manual" or (kind == "srt" and uploaded(arg))):
        raise HTTPException(403, "telemetry here is manual or an uploaded DJI .SRT file")
    if not 0 < spec.rate_hz <= PUBLIC_MAX_RATE_HZ:
        raise HTTPException(422, f"rate must be above 0 and at most {PUBLIC_MAX_RATE_HZ:g} Hz")


def _size(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def _lan_ips() -> list[str]:
    """Addresses other devices (a drone controller) can use to reach this machine."""
    if os.environ.get("PHAROS_IN_DOCKER"):
        return []  # the container cannot see the host's network addresses
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))
