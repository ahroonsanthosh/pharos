"""Flight recordings: camera intrinsics, frames with telemetry, optional ground truth.

The simulator writes exactly the format a real flight is converted to, so the
localizer cannot tell them apart::

    recording/
      camera.json      {"width", "height", "fx", "fy", "cx", "cy"}
      telemetry.jsonl  one line per frame to localize:
                       {"t", "image": "frames/000000.jpg"}  or  {"t", "frame": <video index>}
                       plus "altitude_m", "yaw_deg" (null if no compass),
                       "pitch_deg", "roll_deg" (camera tilt from nadir),
                       "above_takeoff": true if altitude_m is above takeoff
      video.mp4        only when telemetry uses "frame" indices
      truth.jsonl      optional {"t", "x", "y", "altitude_m", "heading_deg", ...}
      takeoff.json     optional {"x", "y"}: where the aircraft took off (map coordinates)

Frames are decoded lazily, so long videos stream from disk.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2

from pharos.geometry import Camera
from pharos.types import Frame, GroundTruth, Telemetry


@dataclass
class Recording:
    camera: Camera
    frames: Sequence[Frame] | _DiskFrames
    truth: list[GroundTruth] | None = None
    takeoff: tuple[float, float] | None = None  # (x, y) in map coordinates, if known

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)


def save_recording(rec: Recording, path: str | Path, jpeg_quality: int = 92) -> None:
    root = Path(path)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    c = rec.camera
    (root / "camera.json").write_text(
        json.dumps(
            {"width": c.width, "height": c.height, "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy},
            indent=2,
        )
    )
    with open(root / "telemetry.jsonl", "w") as f:
        for i, frame in enumerate(rec):
            name = f"frames/{i:06d}.jpg"
            cv2.imwrite(str(root / name), frame.image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            f.write(json.dumps({"t": frame.t, "image": name, **_telemetry_to_json(frame)}) + "\n")
    if rec.truth is not None:
        with open(root / "truth.jsonl", "w") as f:
            for g in rec.truth:
                f.write(json.dumps(_truth_to_json(g)) + "\n")
    if rec.takeoff is not None:
        x, y = rec.takeoff
        (root / "takeoff.json").write_text(json.dumps({"x": x, "y": y}))


def load_recording(path: str | Path) -> Recording:
    root = Path(path)
    camera = Camera(**json.loads((root / "camera.json").read_text()))
    rows = _read_jsonl(root / "telemetry.jsonl")
    truth = None
    if (root / "truth.jsonl").exists():
        truth = [_truth_from_json(r) for r in _read_jsonl(root / "truth.jsonl")]
    takeoff = None
    if (root / "takeoff.json").exists():
        t = json.loads((root / "takeoff.json").read_text())
        takeoff = (float(t["x"]), float(t["y"]))
    return Recording(camera, _DiskFrames(root, rows), truth, takeoff)


class _DiskFrames:
    """Re-iterable lazy frame sequence backed by images or a video file."""

    def __init__(self, root: Path, rows: list[dict]):
        self.root = root
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Frame]:
        if self.rows and "frame" in self.rows[0]:
            yield from self._from_video()
            return
        for r in self.rows:
            image = cv2.imread(str(self.root / r["image"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(self.root / r["image"])
            yield Frame(r["t"], image, _telemetry_from_json(r))

    def _from_video(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.root / "video.mp4"))
        wanted = {r["frame"]: r for r in self.rows}
        index = 0
        try:
            while wanted:
                ok, image = cap.read()
                if not ok:
                    break
                if index in wanted:
                    r = wanted.pop(index)
                    yield Frame(r["t"], image, _telemetry_from_json(r))
                index += 1
        finally:
            cap.release()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _deg(v: float | None) -> float | None:
    return None if v is None else math.degrees(v)


def _rad(v: float | None) -> float | None:
    return None if v is None else math.radians(v)


def _telemetry_to_json(frame: Frame) -> dict:
    tel = frame.telemetry
    return {
        "altitude_m": tel.altitude,
        "yaw_deg": _deg(tel.yaw),
        "pitch_deg": math.degrees(tel.pitch),
        "roll_deg": math.degrees(tel.roll),
    } | ({"above_takeoff": True} if tel.above_takeoff else {})


def _telemetry_from_json(r: dict) -> Telemetry:
    return Telemetry(
        altitude=float(r["altitude_m"]),
        yaw=_rad(r.get("yaw_deg")),
        pitch=math.radians(r.get("pitch_deg", 0.0)),
        roll=math.radians(r.get("roll_deg", 0.0)),
        above_takeoff=bool(r.get("above_takeoff", False)),
    )


def _truth_to_json(g: GroundTruth) -> dict:
    return {
        "t": g.t,
        "x": g.x,
        "y": g.y,
        "altitude_m": g.altitude,
        "heading_deg": math.degrees(g.heading),
        "pitch_deg": math.degrees(g.pitch),
        "roll_deg": math.degrees(g.roll),
    }


def _truth_from_json(r: dict) -> GroundTruth:
    return GroundTruth(
        t=r["t"],
        x=r["x"],
        y=r["y"],
        altitude=r["altitude_m"],
        heading=math.radians(r["heading_deg"]),
        pitch=math.radians(r.get("pitch_deg", 0.0)),
        roll=math.radians(r.get("roll_deg", 0.0)),
    )
