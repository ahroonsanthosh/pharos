import math

import cv2
import pytest

from pharos.sim.flight import simulate
from pharos.sim.scenario import Scenario
from pharos.sim.trajectory import TrajectorySpec
from pharos.sim.world import WorldSpec


@pytest.fixture(scope="session")
def small_sim():
    """A 700 m world and a 30 s flight: fast enough for unit tests."""
    sc = Scenario(
        seed=42,
        world=WorldSpec(size_m=700, villages=2, buildings_per_village=20),
        trajectory=TrajectorySpec(duration_s=30, margin_m=170, altitude=170),
    )
    return simulate(sc)


@pytest.fixture(scope="session")
def tilted_sim():
    """Forward-tilted camera: the matched patch sits ~70 m ahead of the aircraft."""
    sc = Scenario(
        seed=43,
        world=WorldSpec(size_m=700, villages=2, buildings_per_village=20),
        trajectory=TrajectorySpec(duration_s=5, margin_m=200, altitude=150, tilt_deg=25),
    )
    return simulate(sc)


def srt_time(s: float) -> str:
    h, m, sec, ms = int(s // 3600), int(s % 3600 // 60), int(s % 60), int(s % 1 * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def write_dji_recording(rec, geomap, folder):
    """A simulated flight as a DJI pilot would bring it home: an MP4 and its SRT."""
    video = folder / "DJI_0001.MP4"
    fps = 2.0
    h, w = rec.frames[0].image.shape[:2]
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    lines = []
    for i, (frame, truth) in enumerate(zip(rec.frames, rec.truth, strict=True)):
        writer.write(frame.image)
        tel = frame.telemetry
        lat, lon = geomap.to_latlon(truth.x, truth.y)
        lines += [
            str(i + 1),
            f"{srt_time(i / fps)} --> {srt_time((i + 1) / fps)}",
            f"[latitude: {lat:.7f}] [longitude: {lon:.7f}] [rel_alt: {tel.altitude:.2f} abs_alt: 0]"
            f" [gb_yaw: {math.degrees(tel.yaw):.1f} gb_pitch: {math.degrees(tel.pitch) - 90:.1f}"
            f" gb_roll: {math.degrees(tel.roll):.1f}]",
            "",
        ]
    writer.release()
    srt = folder / "DJI_0001.SRT"
    srt.write_text("\n".join(lines))
    return video, srt
