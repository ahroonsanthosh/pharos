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
    """From the second fix (two give the velocity) there is a position, and GNSS is
    checked against it."""
    geomap, rec = small_sim
    store = MapStore(tmp_path / "maps")
    store.add("test-area", geomap)
    video, srt = write_dji_recording(rec, geomap, tmp_path)
    spec = FeedSpec(
        source=str(video), map="test-area", telemetry=f"srt:{srt}", start_from_gnss=True
    )
    feed = Feed("feed-gnss", spec, store)
    assert wait_for(feed) == "finished", feed.error
    assert all(r["status"] in ("tracking", "coasting") for r in feed.records[1:])
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
        demos = {d["name"]: d for d in client.get("/api/scenarios").json()}
        assert demos["nominal"]["group"] == "Demo flights" and demos["nominal"]["about"]
        assert demos["kitchen_sink"]["title"] == "Everything at once"

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


def test_a_public_server_runs_demos_and_uploads_only(small_sim, tmp_path):
    app = create_app(tmp_path, public=True)
    app.state.store.add("area", small_sim[0])
    client = TestClient(app)
    info = client.get("/api/info").json()
    assert info["public"] and "data_dir" not in info
    refused = [
        {"source": "/etc/hostname", "map": "x"},  # a path on the server
        {"source": "rtsp://10.0.0.1/stream", "map": "x"},  # a URL the server would fetch
        {"source": "0", "map": "x"},  # a camera attached to the server
        {"source": f"demo:{tmp_path / 'any.yaml'}"},  # a scenario file, not a built-in one
        {"source": "browser", "map": "x", "telemetry": "mavlink:udpin:0.0.0.0:14551"},
        {"source": "browser", "map": "x", "telemetry": "srt:/etc/hostname"},
    ]
    for body in refused:
        assert client.post("/api/feeds", json=body).status_code == 403, body
    path = client.post("/api/uploads", files={"files": ("clip.mp4", b"not a video")}).json()["path"]
    assert client.post("/api/feeds", json={"source": path, "map": "area"}).status_code == 200
    too_fast = {"source": "browser", "map": "area", "rate_hz": 30}
    assert client.post("/api/feeds", json=too_fast).status_code == 422
    for f in app.state.feeds.values():
        f.stop()


def test_a_public_server_limits_running_feeds(small_sim, tmp_path):
    app = create_app(tmp_path, public=True)
    app.state.store.add("area", small_sim[0])
    client = TestClient(app)
    feed = {"source": "browser", "map": "area"}
    for _ in range(2):  # browser feeds wait for frames, so they keep running
        assert client.post("/api/feeds", json=feed).status_code == 200
    assert client.post("/api/feeds", json=feed).status_code == 429
    for f in app.state.feeds.values():
        f.stop()


@pytest.mark.parametrize("public", [False, True])
def test_clearing_feeds(small_sim, tmp_path, public):
    app = create_app(tmp_path, public=public)
    app.state.store.add("area", small_sim[0])
    client = TestClient(app)
    live = client.post("/api/feeds", json={"source": "browser", "map": "area"}).json()["id"]
    path = client.post("/api/uploads", files={"files": ("clip.mp4", b"not a video")}).json()["path"]
    done = client.post("/api/feeds", json={"source": path, "map": "area"}).json()["id"]
    assert wait_for(app.state.feeds[done]) == "error"
    assert client.delete("/api/feeds").json()["cleared"] == [done]  # finished ones only
    cleared = client.delete("/api/feeds?running=true").json()["cleared"]
    # a public server never stops a running feed that may be someone else's
    assert cleared == ([] if public else [live])
    for f in app.state.feeds.values():
        f.stop()


def test_a_feed_that_cannot_run_is_refused_in_plain_words(small_sim, tmp_path):
    app = create_app(tmp_path)
    app.state.store.add("area", small_sim[0])
    client = TestClient(app)
    refused = {
        "choose a map": {"source": "browser"},
        "no map called": {"source": "browser", "map": "nope"},
        "cannot find the video": {"source": str(tmp_path / "none.mp4"), "map": "area"},
        "rate must be": {"source": "browser", "map": "area", "rate_hz": 0},
        "must be a number": {"source": "browser", "map": "area", "rate_hz": "fast"},
        "height above ground": {"source": "browser", "map": "area", "altitude_m": -50},
        "field of view": {"source": "browser", "map": "area", "hfov_deg": 0},
        "subtitle file": {"source": "browser", "map": "area", "telemetry": "srt:/no.srt"},
        "unknown telemetry": {"source": "browser", "map": "area", "telemetry": "pigeon"},
        "no demo flight": {"source": "demo:nope"},
        "lens distortion is up to 5": {
            "source": "browser",
            "map": "area",
            "lens": [1, 2, 3, 4, 5, 6],
        },
        "not a real lens": {"source": "browser", "map": "area", "lens": [-40]},
    }
    for words, body in refused.items():
        r = client.post("/api/feeds", json=body)
        assert r.status_code == 422 and words in r.json()["detail"], (body, r.text)
    lensed = {"source": "browser", "map": "area", "lens": [-0.2, 0.05]}
    assert client.post("/api/feeds", json=lensed).status_code == 200
    live = client.post("/api/feeds", json={"source": "browser", "map": "area"}).json()["id"]
    for bad in ({"altitude_m": "high"}, {"altitude_m": 0}, {"tilt_deg": 120}):
        assert client.patch(f"/api/feeds/{live}/telemetry", json=bad).status_code == 422
    assert client.patch(f"/api/feeds/{live}/telemetry", json={"altitude_m": 90}).status_code == 200
    for f in app.state.feeds.values():
        f.stop()


def test_a_live_feed_without_video_says_what_to_check(small_sim, tmp_path, monkeypatch):
    monkeypatch.setattr("pharos.server.feeds.NO_VIDEO_S", 0.5)
    store = MapStore(tmp_path / "maps")
    store.add("area", small_sim[0])
    feed = Feed("feed-1", FeedSpec(source="push:drone", map="area"), store)
    deadline = time.monotonic() + 20
    while not feed.error and time.monotonic() < deadline:
        time.sleep(0.2)
    assert feed.state == "running" and "MediaMTX" in (feed.error or "")
    feed.stop()
