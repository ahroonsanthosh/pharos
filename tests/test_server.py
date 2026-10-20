"""Service: maps, feeds from recorded video + DJI SRT, browser frames, HTTP API."""

import time

import cv2
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pharos.server.app import create_app
from pharos.server.feeds import Feed, FeedSpec
from pharos.server.store import MapStore
from tests.conftest import write_dji_recording


def wait_for(feed, states=("finished", "error"), timeout=120):
    deadline = time.monotonic() + timeout
    while feed.state not in states and time.monotonic() < deadline:
        time.sleep(0.2)
    return feed.state


def test_recorded_dji_flight(small_sim, tmp_path):
    geomap, rec = small_sim
    store = MapStore(tmp_path / "maps")
    store.add("test-area", geomap)
    video, srt = write_dji_recording(rec, geomap, tmp_path)
    spec = FeedSpec(source=str(video), map="test-area", hfov_deg=70.0, telemetry=f"srt:{srt}")
    feed = Feed("feed-1", spec, store)
    assert wait_for(feed) == "finished", feed.error
    assert len(feed.records) == len(rec.frames)
    tracking = [r for r in feed.records if r["status"] == "tracking"]
    assert len(tracking) > len(feed.records) / 2
    errors = sorted(r["error_m"] for r in tracking)
    assert errors[len(errors) // 2] < 8  # median error against the SRT GPS
    assert "lat" in tracking[-1] and feed.preview is not None


def test_a_feed_can_start_from_the_first_gnss_fix(small_sim, tmp_path):
    """From the first fix there is a position at once, and GNSS checked against it."""
    geomap, rec = small_sim
    store = MapStore(tmp_path / "maps")
    store.add("test-area", geomap)
    video, srt = write_dji_recording(rec, geomap, tmp_path)
    spec = FeedSpec(
        source=str(video), map="test-area", telemetry=f"srt:{srt}", start_from_gnss=True
    )
    feed = Feed("feed-gnss", spec, store)
    assert wait_for(feed) == "finished", feed.error
    assert all(r["status"] in ("tracking", "coasting") for r in feed.records)
    checks = {r.get("gnss_check") for r in feed.records if r["status"] == "tracking"}
    assert checks == {"agrees"}


def test_browser_frames(small_sim, tmp_path):
    geomap, rec = small_sim
    store = MapStore(tmp_path / "maps")
    store.add("test-area", geomap)
    spec = FeedSpec(source="browser", map="test-area", hfov_deg=70.0, altitude_m=170, rate_hz=50)
    feed = Feed("feed-2", spec, store)
    # Live sources drop frames that arrive while one is processing, so keep sending.
    deadline = time.monotonic() + 60
    for frame in rec.frames:
        if len(feed.records) >= 2 or time.monotonic() > deadline:
            break
        feed.pushed.put(cv2.imencode(".jpg", frame.image)[1].tobytes())
        time.sleep(0.4)
    feed.stop()
    assert feed.state == "stopped" and len(feed.records) >= 2


def test_bad_source_reports_error(tmp_path):
    store = MapStore(tmp_path / "maps")
    feed = Feed("feed-3", FeedSpec(source="no-such-file.mp4", map="missing"), store)
    assert wait_for(feed) == "error" and "missing" in feed.error


def test_http_api(small_sim, tmp_path):
    geomap, _ = small_sim
    app = create_app(tmp_path)
    app.state.store.add("area", geomap)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        maps = client.get("/api/maps").json()
        assert maps[0]["name"] == "area" and maps[0]["gsd"] == 1.0
        assert client.get("/api/maps/area/preview.jpg").headers["content-type"] == "image/jpeg"
        assert "nominal" in client.get("/api/scenarios").json()

        r = client.post("/api/feeds", json={"source": "browser", "map": "area"})
        feed_id = r.json()["id"]
        assert client.get("/api/feeds").json()[0]["id"] == feed_id
        r = client.patch(f"/api/feeds/{feed_id}/telemetry", json={"altitude_m": 150, "tilt_deg": 5})
        assert r.status_code == 200
        with client.websocket_connect(f"/api/feeds/{feed_id}/frames") as ws:
            ws.send_bytes(cv2.imencode(".jpg", geomap.image[:480, :640])[1].tobytes())
        deadline = time.monotonic() + 30
        while not client.get(f"/api/feeds/{feed_id}/records").json()["records"]:
            assert time.monotonic() < deadline, "browser frame was not processed"
            time.sleep(0.2)
        assert client.get(f"/api/feeds/{feed_id}/preview.jpg").status_code == 200
        assert client.post("/api/feeds", json={"source": "x", "bogus": 1}).status_code == 422
        assert client.delete(f"/api/feeds/{feed_id}").status_code == 200
        assert client.get("/api/feeds").json() == []
