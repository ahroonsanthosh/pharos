"""Feeds: any video source plus a telemetry source, localized frame by frame.

``FeedSpec.source`` selects where frames come from:

* ``demo:<scenario>``   a simulated flight played back in real time
* ``browser``           frames sent by the dashboard (webcam, capture card, screen)
* ``push:<path>``       a stream pushed to the bundled MediaMTX server
                        (RTMP from a DJI controller or OBS, SRT/RTSP from a ground station)
* ``rtsp://``, ``rtmp://``, ``udp://``, ``srt://``, ``http(s)://...m3u8``  live stream URLs
* ``0``, ``1``, ...     a camera or capture card attached to this machine
* a folder             photos with EXIF time/GPS (DJI and senseFly metadata understood)
* anything else         a video file path (processed at ``rate_hz`` of video time)

Live sources always hand over the newest frame and drop the rest, so a slow
machine falls behind gracefully instead of lagging further and further.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from pharos.eval.export import record
from pharos.footage import load_photos
from pharos.geomap import GeoMap
from pharos.geometry import Camera
from pharos.nav.gnss import GnssCheck
from pharos.nav.localizer import Localizer
from pharos.server.store import MapStore
from pharos.sim.flight import simulate
from pharos.sim.scenario import load_scenario
from pharos.telemetry import (
    ManualTelemetry,
    MavlinkTelemetry,
    SrtTelemetry,
    TelemetrySample,
    TelemetrySource,
)
from pharos.types import Frame

log = logging.getLogger(__name__)

LIVE_SCHEMES = ("rtsp", "rtsps", "rtmp", "rtmps", "srt", "udp", "tcp")
MAX_WIDTH = 960  # frames are downscaled to this width before localization
PREVIEW_WIDTH = 960  # the dashboard can show the camera large
NO_VIDEO_S = 10.0  # a live feed this long without a frame says what to check


@dataclass(frozen=True)
class FeedSpec:
    source: str
    map: str = ""  # map name in the store (demo feeds bring their own)
    name: str = ""
    hfov_deg: float = 70.0  # horizontal field of view of the video as recorded
    rate_hz: float = 2.0
    telemetry: str = "manual"  # "manual", "mavlink[:<endpoint>]" or "srt:<path>"
    altitude_m: float = 100.0  # manual telemetry: height above ground
    tilt_deg: float = 0.0  # manual telemetry: camera tilt forward from straight down
    mount: str = "gimbal"  # MAVLink: "gimbal" (stabilized) or "fixed" to the airframe
    video_delay_s: float = 0.0  # live feeds: how far the video lags the telemetry
    # Begin from the aircraft's first GNSS fix (then never use GNSS again, so jamming or
    # spoofing later cannot pull the track away); off: a cold start, GNSS only shown.
    start_from_gnss: bool = False
    # Lens distortion (k1, k2, p1, p2, k3), or fewer, as ``pharos lens`` measures it.
    lens: tuple[float, ...] = ()


def check_spec(spec: FeedSpec, maps: list[str], scenarios: list[str]) -> None:
    """Raise ValueError, in plain words, for a feed that could not run as asked."""
    for name in ("hfov_deg", "rate_hz", "altitude_m", "tilt_deg", "video_delay_s"):
        value = getattr(spec, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a number, not {value!r}")
    for name in ("source", "map", "name", "telemetry", "mount"):
        if not isinstance(getattr(spec, name), str):
            raise ValueError(f"{name} must be text")
    lens = spec.lens if isinstance(spec.lens, tuple | list) else None
    if (
        lens is None
        or len(lens) > 5
        or not all(
            isinstance(k, int | float) and not isinstance(k, bool) and math.isfinite(k)
            for k in lens
        )
    ):
        raise ValueError("lens distortion is up to 5 numbers: k1, k2, p1, p2, k3")
    if any(abs(k) > 10 for k in lens):
        raise ValueError("lens distortion coefficients that large are not a real lens")
    if not isinstance(spec.start_from_gnss, bool):
        raise ValueError("start_from_gnss must be true or false")
    if not spec.source.strip():
        raise ValueError("choose a video source")
    if spec.source.startswith("demo:"):
        demo = spec.source.removeprefix("demo:")
        if demo not in scenarios and not Path(demo).is_file():
            raise ValueError(f"there is no demo flight called {demo!r}")
        return  # a demo brings its own map, camera and telemetry
    if not spec.map:
        raise ValueError("choose a map, or upload one for the area you are flying")
    if spec.map not in maps:
        raise ValueError(f"there is no map called {spec.map!r}; upload it first")
    if not 0 < spec.rate_hz <= 30:
        raise ValueError("rate must be above 0 and at most 30 frames a second")
    if not 1 <= spec.hfov_deg <= 179:
        raise ValueError("horizontal field of view must be between 1 and 179 degrees")
    if not -90 <= spec.tilt_deg <= 90:
        raise ValueError("camera tilt must be between -90 and 90 degrees")
    if spec.video_delay_s < 0:
        raise ValueError("video delay cannot be negative")
    if spec.mount not in ("gimbal", "fixed"):
        raise ValueError("camera mount is gimbal or fixed")
    kind, _, arg = spec.telemetry.partition(":")
    if kind == "manual" and spec.altitude_m <= 0:
        raise ValueError("height above ground must be more than 0 m")
    if kind == "srt" and not Path(arg).is_file():
        raise ValueError(f"cannot find the subtitle file {arg!r}")
    if kind not in ("manual", "mavlink", "srt"):
        raise ValueError(f"unknown telemetry {spec.telemetry!r}; use manual, mavlink or srt")
    live = (
        spec.source in ("browser",)
        or spec.source.startswith("push:")
        or spec.source.isdigit()
        or "://" in spec.source
    )
    if not live and not Path(spec.source).exists():
        raise ValueError(f"cannot find the video or photos at {spec.source!r}")


def no_video_hint(source: str) -> str:
    """What to check when a live source has sent nothing."""
    if source.startswith("push:"):
        path = source.removeprefix("push:")
        return (
            f"No video yet. Check the drone or app is streaming to rtmp://<this computer>:1935/"
            f"{path}, and that MediaMTX is running (Docker starts it for you)."
        )
    if source == "browser":
        return "No video yet from the browser camera. Allow camera access, or pick another camera."
    if source.isdigit():
        return (
            f"No video yet from camera {source}. Check it is plugged in and not in use elsewhere."
        )
    return f"No video yet from {source}. Check the address, and that the stream is on."


class FrameSource(Protocol):
    live: bool

    def read(self, timeout: float) -> tuple[float, np.ndarray] | None:
        """Next frame as (seconds, BGR image); None on timeout. Raises EOFError at the end."""
        ...

    def close(self) -> None: ...


class Feed:
    def __init__(self, feed_id: str, spec: FeedSpec, store: MapStore):
        self.id = feed_id
        self.spec = spec
        self.store = store
        self.state = "starting"
        self.error: str | None = None
        self.map_name = spec.map
        self.records: list[dict] = []
        self.preview: bytes | None = None
        self.telemetry: TelemetrySource | None = None
        self.camera: Camera | None = None  # known intrinsics; else derived from hfov_deg
        self.pushed: PushedFrames | None = PushedFrames() if spec.source == "browser" else None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=feed_id, daemon=True)
        self._thread.start()

    def summary(self) -> dict:
        last = self.records[-1] if self.records else None
        return {
            "id": self.id,
            "name": self.spec.name or self.spec.source,
            "spec": asdict(self.spec),
            "map": self.map_name,
            "state": self.state,
            "error": self.error,
            "frames": len(self.records),
            "last": last,
        }

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    # -- worker -----------------------------------------------------------

    def _run(self) -> None:
        source = None
        try:
            source, geomap = self._open()
            self.state = "running"
            self._loop(source, geomap)
            self.state = "stopped" if self._stop.is_set() else "finished"
        except Exception as exc:  # report any failure to the dashboard
            log.exception("feed %s failed", self.id)
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            if source is not None:
                source.close()
            if self.telemetry is not None:
                self.telemetry.close()

    def _open(self) -> tuple[FrameSource, GeoMap]:
        spec = self.spec
        if spec.source.startswith("demo:"):
            demo = DemoSource(spec.source.removeprefix("demo:"), self.store)
            self.map_name, self.telemetry, self.camera = demo.map_name, demo, demo.camera
            return demo, self.store.get(demo.map_name)
        geomap = self.store.get(spec.map)
        if Path(spec.source).is_dir():
            photos = PhotoSource(Path(spec.source), spec)
            self.telemetry, self.camera = photos, photos.camera
            return photos, geomap
        self.telemetry = _open_telemetry(spec)
        if self.pushed is not None:
            return self.pushed, geomap
        return _open_capture(spec.source, spec.rate_hz), geomap

    def _loop(self, source: FrameSource, geomap: GeoMap) -> None:
        assert self.telemetry is not None
        localizer: Localizer | None = None
        check, started, last_fix = GnssCheck(), False, None
        period = 1.0 / self.spec.rate_hz
        next_due = last_frame = time.monotonic()
        while not self._stop.is_set():
            if source.live:
                time.sleep(max(0.0, next_due - time.monotonic()))
            try:
                item = source.read(timeout=1.0)
            except EOFError:
                return
            if item is None:
                if source.live and time.monotonic() - last_frame > NO_VIDEO_S:
                    self.error = no_video_hint(self.spec.source)
                continue
            last_frame = time.monotonic()
            self.error = None
            next_due = time.monotonic() + period
            t, image = item
            sample = self.telemetry.at(t - self.spec.video_delay_s if source.live else t)
            if sample is None:
                self.error = "waiting for telemetry"
                continue
            self.error = None
            image = _limit_width(image, MAX_WIDTH)
            if localizer is None:
                h, w = image.shape[:2]
                camera = self.camera or Camera.from_fov(w, h, self.spec.hfov_deg)
                if self.spec.lens and not self.spec.source.startswith("demo:"):
                    camera = replace(camera, dist=tuple(self.spec.lens))
                if sample.telemetry.above_takeoff and self.map_name:
                    geomap = self.store.with_terrain(self.map_name)
                localizer = Localizer(geomap, camera)
            home = self.telemetry.takeoff  # a MAVLink home position may come later
            xy = geomap.from_latlon(*home) if home is not None else None
            if localizer.takeoff_elevation is None and xy is not None:
                localizer.set_takeoff(xy)
            gnss_xy = geomap.from_latlon(*sample.gnss) if sample.gnss is not None else None
            if self.spec.start_from_gnss and not started and gnss_xy is not None:
                if last_fix is not None and t > last_fix[0]:  # two fixes give the velocity
                    velocity = (np.asarray(gnss_xy) - last_fix[1]) / (t - last_fix[0])
                    localizer.start_at(gnss_xy, check.gnss_sigma_m, velocity=tuple(velocity))
                    started = True
                last_fix = (t, np.asarray(gnss_xy))
            estimate = localizer.process(Frame(t, image, sample.telemetry))
            used = localizer.log[-1]["tel"]
            rec = record(estimate, sample.telemetry, sample.gnss, geomap, used)
            if (verdict := check.update(estimate, gnss_xy)) is not None:
                rec["gnss_check"] = verdict  # "disagrees": the GNSS may be spoofed
            self.records.append(rec)
            preview = cv2.imencode(".jpg", _limit_width(image, PREVIEW_WIDTH))[1]
            self.preview = preview.tobytes()


# -- sources -----------------------------------------------------------------


def _open_capture(source: str, rate_hz: float) -> FrameSource:
    if source.startswith("push:"):
        host = os.environ.get("PHAROS_MEDIAMTX_HOST", "localhost")
        return LiveCapture(f"rtsp://{host}:8554/{source.removeprefix('push:')}")
    if source.isdigit():
        return LiveCapture(int(source))
    scheme = source.split("://", 1)[0].lower() if "://" in source else ""
    if scheme in LIVE_SCHEMES or (scheme.startswith("http") and ".m3u8" in source):
        return LiveCapture(source)
    if not scheme and not Path(source).exists():
        raise FileNotFoundError(source)
    return FileCapture(source, rate_hz)


class LiveCapture:
    """Keeps only the newest frame of a live stream or device; reconnects if it drops."""

    live = True

    def __init__(self, target: str | int):
        self.target = target
        self._cond = threading.Condition()
        self._frame: tuple[float, np.ndarray] | None = None
        self._seq = self._taken = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._grab, name="capture", daemon=True)
        self._thread.start()

    def _grab(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.target)
            while cap.isOpened() and not self._stop.is_set():
                ok, image = cap.read()
                if not ok:
                    break
                with self._cond:
                    self._frame = (time.monotonic(), image)
                    self._seq += 1
                    self._cond.notify_all()
            cap.release()
            self._stop.wait(1.0)  # stream not up yet or dropped: retry

    def read(self, timeout: float) -> tuple[float, np.ndarray] | None:
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._taken, timeout):
                return None
            self._taken = self._seq
            return self._frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)


class FileCapture:
    """Steps through a recorded video at ``rate_hz`` of video time, as fast as possible."""

    live = False

    def __init__(self, path: str, rate_hz: float):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise OSError(f"cannot open video {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.stride = max(round(fps / rate_hz), 1)
        self.fps = fps
        self.index = 0

    def read(self, timeout: float) -> tuple[float, np.ndarray]:
        for _ in range(self.stride - 1):
            if not self.cap.grab():
                raise EOFError
            self.index += 1
        ok, image = self.cap.read()
        if not ok:
            raise EOFError
        t = self.index / self.fps
        self.index += 1
        return t, image

    def close(self) -> None:
        self.cap.release()


class PushedFrames:
    """Frames sent over a WebSocket by the dashboard (JPEG bytes)."""

    live = True

    def __init__(self):
        self._cond = threading.Condition()
        self._frame: tuple[float, np.ndarray] | None = None
        self._seq = self._taken = 0

    def put(self, jpeg: bytes) -> None:
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        with self._cond:
            self._frame = (time.monotonic(), image)
            self._seq += 1
            self._cond.notify_all()

    def read(self, timeout: float) -> tuple[float, np.ndarray] | None:
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._taken, timeout):
                return None
            self._taken = self._seq
            return self._frame

    def close(self) -> None:
        pass


class PhotoSource:
    """A folder of photos, played as fast as possible with each photo's own metadata."""

    live = False

    def __init__(self, folder: Path, spec: FeedSpec):
        try:
            footage = load_photos(folder)
        except ValueError:  # photos lack height or field of view: use the feed's settings
            footage = load_photos(folder, hfov_deg=spec.hfov_deg, altitude_m=spec.altitude_m)
        self.camera = footage.recording.camera
        self.frames = iter(footage.recording)
        self.gnss = footage.gnss
        self.takeoff = footage.takeoff
        self.samples: dict[float, TelemetrySample] = {}

    def read(self, timeout: float) -> tuple[float, np.ndarray]:
        frame = next(self.frames, None)
        if frame is None:
            raise EOFError
        self.samples[frame.t] = TelemetrySample(frame.telemetry, self.gnss[len(self.samples)])
        return frame.t, frame.image

    def at(self, t: float) -> TelemetrySample | None:
        return self.samples.get(t)

    def close(self) -> None:
        pass


class DemoSource:
    """A simulated flight, played back in real time with its telemetry and GNSS truth."""

    live = False

    def __init__(self, scenario: str, store: MapStore):
        sc = load_scenario(scenario)
        geomap, rec = simulate(sc)
        self.map_name = store.add(f"demo-{sc.name}", geomap)
        self.camera = rec.camera
        self.frames = list(rec.frames)
        self.takeoff = geomap.to_latlon(*rec.takeoff) if rec.takeoff else None
        truth = rec.truth or []
        self.samples = {
            f.t: TelemetrySample(f.telemetry, geomap.to_latlon(g.x, g.y))
            for f, g in zip(self.frames, truth, strict=True)
        }
        self.index = 0
        self.start = time.monotonic()

    def read(self, timeout: float) -> tuple[float, np.ndarray]:
        if self.index >= len(self.frames):
            raise EOFError
        frame = self.frames[self.index]
        time.sleep(max(0.0, self.start + frame.t - time.monotonic()))
        self.index += 1
        return frame.t, frame.image

    def at(self, t: float) -> TelemetrySample | None:
        return self.samples.get(t)

    def close(self) -> None:
        pass


def _open_telemetry(spec: FeedSpec) -> TelemetrySource:
    kind, _, arg = spec.telemetry.partition(":")
    if kind == "manual":
        return ManualTelemetry(spec.altitude_m, math.radians(spec.tilt_deg))
    if kind == "mavlink":
        return MavlinkTelemetry(arg or "udpin:0.0.0.0:14551", mount=spec.mount)
    if kind == "srt":
        return SrtTelemetry(arg)
    raise ValueError(f"unknown telemetry {spec.telemetry!r}")


# -- helpers -----------------------------------------------------------------


def _limit_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= width:
        return image
    return cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
