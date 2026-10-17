"""Recorded footage for offline analysis: a video file or a folder of photos.

Photos carry their own metadata: capture time and GPS from EXIF, plus vendor
XMP fields: DJI records height above takeoff and gimbal angles, senseFly
(eBee) records the camera's attitude and compass heading.
GPS positions are never given to the localizer; they are only used to score
its output.

Height above ground matters (it sets the scale), and photo GPS altitude is a
trap: the EXIF standard says "above sea level", but many aircraft write the
height above the WGS84 ellipsoid, which differs by tens of metres. Prefer, in
order: an explicit ``altitude_m``, DJI's relative altitude, or GPS altitude
minus ``ground_m`` given in the same vertical reference as the photos.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image

from pharos.geometry import Camera
from pharos.recording import Recording
from pharos.telemetry import SrtTelemetry, TelemetrySample
from pharos.types import Frame, Telemetry

PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
MAX_WIDTH = 1280


@dataclass
class Footage:
    recording: Recording
    gnss: list[tuple[float, float] | None]  # aircraft-reported position per frame


@dataclass(frozen=True)
class PhotoInfo:
    path: Path
    time: datetime | None
    gnss: tuple[float, float] | None
    gps_altitude: float | None
    relative_altitude: float | None  # DJI: height above the takeoff point
    gimbal_pitch: float | None  # degrees, -90 = straight down
    gimbal_roll: float | None
    tilt: tuple[float, float] | None  # (forward, right) degrees from nadir, if recorded
    yaw: float | None  # degrees, compass
    focal_35mm: float | None
    size: tuple[int, int]


def load_footage(
    path: str | Path,
    *,
    hfov_deg: float | None = None,
    altitude_m: float | None = None,
    ground_m: float | None = None,
    srt: str | Path | None = None,
    rate_hz: float = 2.0,
) -> Footage:
    path = Path(path)
    if path.is_dir():
        return load_photos(path, hfov_deg=hfov_deg, altitude_m=altitude_m, ground_m=ground_m)
    return load_video(path, hfov_deg=hfov_deg, altitude_m=altitude_m, srt=srt, rate_hz=rate_hz)


# -- photos ------------------------------------------------------------------


def load_photos(
    folder: Path,
    *,
    hfov_deg: float | None = None,
    altitude_m: float | None = None,
    ground_m: float | None = None,
) -> Footage:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
    if not files:
        raise ValueError(f"{folder}: no photos found")
    infos = [read_photo_info(p) for p in files]
    infos.sort(key=lambda i: (i.time or datetime.min, i.path.name))
    t0 = next((i.time for i in infos if i.time), None)

    w, h = infos[0].size
    hfov = hfov_deg or _hfov_from_35mm(infos[0].focal_35mm, w, h)
    if hfov is None:
        raise ValueError("unknown field of view: pass hfov_deg (no 35 mm focal length in EXIF)")
    width = min(w, MAX_WIDTH)
    camera = Camera.from_fov(width, round(h * width / w), hfov)

    telemetry, times = [], []
    for k, info in enumerate(infos):
        telemetry.append(_photo_telemetry(info, altitude_m, ground_m))
        times.append((info.time - t0).total_seconds() if info.time and t0 else float(k))
    frames = _PhotoFrames([i.path for i in infos], times, telemetry, width)
    return Footage(Recording(camera, frames), [i.gnss for i in infos])


def read_photo_info(path: Path) -> PhotoInfo:
    with Image.open(path) as im:
        exif = im.getexif()
        size = im.size
    gps = exif.get_ifd(0x8825)
    sub = exif.get_ifd(0x8769)
    stamp = sub.get(0x9003) or exif.get(0x0132)  # DateTimeOriginal, else DateTime
    when = _parse_time(stamp)
    gnss, gps_alt = None, None
    if 2 in gps and 4 in gps:
        lat = _dms(gps[2]) * (-1 if gps.get(1) == "S" else 1)
        lon = _dms(gps[4]) * (-1 if gps.get(3) == "W" else 1)
        gnss = (lat, lon)
    if 6 in gps:
        gps_alt = float(gps[6]) * (-1 if gps.get(5) in (1, b"\x01") else 1)
    xmp = _xmp_numbers(path)
    yaw = next((xmp[k] for k in _YAW_KEYS if k in xmp), None)
    focal = sub.get(0xA405)
    return PhotoInfo(
        path=path,
        time=when,
        gnss=gnss,
        gps_altitude=gps_alt,
        relative_altitude=xmp.get("drone-dji:RelativeAltitude"),
        gimbal_pitch=xmp.get("drone-dji:GimbalPitchDegree"),
        gimbal_roll=xmp.get("drone-dji:GimbalRollDegree"),
        tilt=_sensefly_tilt(xmp),
        yaw=yaw,
        focal_35mm=float(focal) if focal else None,
        size=size,
    )


def _photo_telemetry(info: PhotoInfo, altitude_m: float | None, ground_m: float | None):
    if altitude_m is not None:
        altitude = altitude_m
    elif info.relative_altitude is not None:
        altitude = info.relative_altitude
    elif info.gps_altitude is not None and ground_m is not None:
        altitude = info.gps_altitude - ground_m
    else:
        raise ValueError(
            f"{info.path.name}: unknown height above ground; pass altitude_m, or ground_m "
            "(ground height in the same reference as the photos' GPS altitude)"
        )
    if info.gimbal_pitch is not None:
        forward, right = 90.0 + info.gimbal_pitch, info.gimbal_roll or 0.0
    else:
        forward, right = info.tilt or (0.0, 0.0)
    return Telemetry(
        altitude=altitude,
        yaw=None if info.yaw is None else math.radians(info.yaw),
        pitch=math.radians(forward),
        roll=math.radians(right),
    )


class _PhotoFrames:
    """Decodes photos one at a time, downscaled to ``width``."""

    def __init__(self, paths: list[Path], times: list[float], telemetry: list, width: int):
        self.paths, self.times, self.telemetry, self.width = paths, times, telemetry, width

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Frame]:
        for path, t, tel in zip(self.paths, self.times, self.telemetry, strict=True):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"cannot read {path}")
            h, w = image.shape[:2]
            if w != self.width:
                size = (self.width, round(h * self.width / w))
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            yield Frame(t, image, tel)


# -- video -------------------------------------------------------------------


def load_video(
    path: Path,
    *,
    hfov_deg: float | None = None,
    altitude_m: float | None = None,
    srt: str | Path | None = None,
    rate_hz: float = 2.0,
) -> Footage:
    if hfov_deg is None:
        raise ValueError("pass hfov_deg for video footage")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    srt_path = Path(srt) if srt else _sibling_srt(path)
    telemetry = SrtTelemetry(srt_path) if srt_path else None
    if telemetry is None and altitude_m is None:
        raise ValueError("unknown height above ground: pass altitude_m or a DJI .SRT file")

    stride = max(round(fps / rate_hz), 1)
    times = [i / fps for i in range(0, count, stride)]
    samples = [
        telemetry.at(t) if telemetry else TelemetrySample(Telemetry(altitude_m)) for t in times
    ]
    width = min(w, MAX_WIDTH)
    camera = Camera.from_fov(width, round(h * width / w), hfov_deg)
    frames = _VideoFrames(path, stride, fps, samples, width)
    return Footage(Recording(camera, frames), [s.gnss if s else None for s in samples])


class _VideoFrames:
    def __init__(self, path: Path, stride: int, fps: float, samples: list, width: int):
        self.path, self.stride, self.fps, self.samples, self.width = (
            path,
            stride,
            fps,
            samples,
            width,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        try:
            for k, sample in enumerate(self.samples):
                if k:
                    for _ in range(self.stride - 1):
                        cap.grab()
                ok, image = cap.read()
                if not ok:
                    return
                h, w = image.shape[:2]
                if w != self.width:
                    size = (self.width, round(h * self.width / w))
                    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                tel = sample.telemetry if sample else Telemetry(0.0)
                yield Frame(k * self.stride / self.fps, image, tel)
        finally:
            cap.release()


# -- helpers -----------------------------------------------------------------

_XMP_FIELD = re.compile(rb'([\w-]+:\w+)="([+-]?[\d.]+)"')
# Compass heading of the camera, by vendor.
_YAW_KEYS = ("drone-dji:GimbalYawDegree", "drone-dji:FlightYawDegree", "Camera:Yaw")


def _sensefly_tilt(xmp: dict[str, float]) -> tuple[float, float] | None:
    """senseFly Camera:Pitch/Roll as (forward, right) tilt from nadir.

    The sign convention is undocumented. It was determined from data: on the
    OpenDroneMap Aukerman survey, pitch as recorded and roll negated is the
    only combination for which visual odometry agrees with the photos' GPS
    (median step error 14 m against 20 m ignoring attitude, 26-29 m otherwise).
    """
    if "Camera:Pitch" not in xmp or "Camera:Roll" not in xmp:
        return None
    return xmp["Camera:Pitch"], -xmp["Camera:Roll"]


def _xmp_numbers(path: Path) -> dict[str, float]:
    """Numeric ``prefix:Name="value"`` attributes from a photo's XMP packet."""
    with open(path, "rb") as f:
        head = f.read(256 * 1024)
    return {k.decode(): float(v) for k, v in _XMP_FIELD.findall(head)}


def _dms(v) -> float:
    return float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600


def _parse_time(stamp) -> datetime | None:
    if not stamp:
        return None
    try:
        return datetime.strptime(str(stamp).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _hfov_from_35mm(focal_35mm: float | None, w: int, h: int) -> float | None:
    """35 mm-equivalent focal lengths are defined on the 43.27 mm full-frame diagonal."""
    if not focal_35mm:
        return None
    half_diag = math.atan(43.27 / (2 * focal_35mm))
    return math.degrees(2 * math.atan(math.tan(half_diag) * w / math.hypot(w, h)))


def _sibling_srt(video: Path) -> Path | None:
    for suffix in (".SRT", ".srt"):
        candidate = video.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None
