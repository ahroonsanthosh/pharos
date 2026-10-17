"""HTTP API and dashboard for live localization (``pharos serve``)."""

from __future__ import annotations

import itertools
import math
import os
import shutil
import socket
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from pharos.config import from_dict
from pharos.server.feeds import Feed, FeedSpec
from pharos.server.store import MapStore, safe_name
from pharos.server.telemetry import ManualTelemetry
from pharos.sim.scenario import builtin_scenarios

STATIC = Path(__file__).parent / "static"


def create_app(data_dir: str | Path) -> FastAPI:
    data = Path(data_dir)
    uploads = data / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    store = MapStore(data / "maps")
    feeds: dict[str, Feed] = {}
    ids = itertools.count(1)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        for feed in feeds.values():
            feed.stop()

    app = FastAPI(title="Pharos", docs_url="/api/docs", lifespan=lifespan)
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
        return {"lan_ips": _lan_ips(), "data_dir": str(data.resolve())}

    # -- maps -------------------------------------------------------------

    @app.get("/api/maps")
    def list_maps() -> list[dict]:
        return [store.info(name) for name in store.names()]

    @app.post("/api/maps")
    def upload_map(name: str, files: list[UploadFile]) -> dict:
        """A GeoTIFF, or a map ``.json`` sidecar together with its image."""
        with tempfile.TemporaryDirectory() as tmp:
            saved = [_save_upload(f, Path(tmp)) for f in files]
            main = next((p for p in saved if p.suffix.lower() in (".tif", ".tiff", ".json")), None)
            if main is None:
                raise HTTPException(422, "upload a GeoTIFF or a .json sidecar with its image")
            try:
                stored = store.add_file(name, main)
            except (ValueError, OSError, KeyError) as exc:
                raise HTTPException(422, str(exc)) from exc
        return store.info(stored)

    @app.get("/api/maps/{name}/preview.jpg")
    def map_preview(name: str) -> Response:
        try:
            return Response(store.preview(name), media_type="image/jpeg")
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/scenarios")
    def scenarios() -> list[str]:
        return builtin_scenarios()

    @app.post("/api/uploads")
    def upload_file(file: UploadFile) -> dict:
        """Store a video or SRT file; returns the server path to use in a feed."""
        return {"path": str(_save_upload(file, uploads))}

    # -- feeds ------------------------------------------------------------

    @app.get("/api/feeds")
    def list_feeds() -> list[dict]:
        return [f.summary() for f in feeds.values()]

    @app.post("/api/feeds")
    def create_feed(body: dict) -> dict:
        try:
            spec = from_dict(FeedSpec, body)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        feed_id = f"feed-{next(ids)}"
        feeds[feed_id] = Feed(feed_id, spec, store)
        return feeds[feed_id].summary()

    @app.delete("/api/feeds/{feed_id}")
    def delete_feed(feed_id: str) -> dict:
        feed = get_feed(feed_id)
        feed.stop()
        del feeds[feed_id]
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
        if "altitude_m" in body:
            feed.telemetry.update(altitude=float(body["altitude_m"]))
        if "tilt_deg" in body:
            feed.telemetry.update(pitch=math.radians(float(body["tilt_deg"])))
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


def _save_upload(upload: UploadFile, folder: Path) -> Path:
    path = folder / safe_name(Path(upload.filename or "upload").name)
    with open(path, "wb") as out:
        shutil.copyfileobj(upload.file, out)
    return path


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
