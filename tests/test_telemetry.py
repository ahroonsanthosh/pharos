import math
import socket
import time

import pytest

from pharos.server.telemetry import MavlinkState, SrtTelemetry, parse_dji_srt

MODERN_SRT = """1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
2024-05-03 10:23:45.123
[iso: 100] [shutter: 1/1000.0] [fnum: 1.7] [ev: 0] [focal_len: 24.00] \
[latitude: 46.951234] [longitude: 7.441234] [rel_alt: 120.500 abs_alt: 640.300] \
[gb_yaw: 35.2 gb_pitch: -90.0 gb_roll: 0.0] </font>

2
00:00:01,000 --> 00:00:01,033
<font size="28">FrameCnt: 31, DiffTime: 33ms
[latitude: 46.951300] [longtitude: 7.441300] [rel_alt: 121.000 abs_alt: 640.800] \
[gb_yaw: 36.0 gb_pitch: -70.0 gb_roll: 1.0] </font>
"""

LEGACY_SRT = """1
00:00:00,000 --> 00:00:01,000
HOME(7.4400,46.9500) 2017.08.05 14:11:51
GPS(7.4412,46.9512,15) ISO:100 Shutter:60 EV:0 Fnum:2.2
BAROMETER:95.3
"""


def test_modern_dji_srt():
    samples = parse_dji_srt(MODERN_SRT)
    assert [t for t, _ in samples] == [0.0, 1.0]
    first, second = samples[0][1], samples[1][1]
    assert first.telemetry.altitude == 120.5
    assert math.isclose(first.telemetry.pitch, 0.0)  # gimbal straight down
    assert math.isclose(math.degrees(first.telemetry.yaw), 35.2)
    assert first.gnss == (46.951234, 7.441234)
    assert math.isclose(math.degrees(second.telemetry.pitch), 20.0)  # tilted forward
    assert second.gnss == (46.9513, 7.4413)  # "longtitude" typo in some firmware


def test_legacy_dji_srt():
    ((t, sample),) = parse_dji_srt(LEGACY_SRT)
    assert t == 0.0
    assert sample.telemetry.altitude == 95.3
    assert sample.gnss == (46.9512, 7.4412)
    assert sample.telemetry.yaw is None


def test_srt_lookup_by_video_time(tmp_path):
    path = tmp_path / "DJI_0001.SRT"
    path.write_text(MODERN_SRT)
    srt = SrtTelemetry(path)
    assert srt.at(0.5).telemetry.altitude == 120.5
    assert srt.at(5.0).telemetry.altitude == 121.0
    assert srt.at(-1.0).telemetry.altitude == 120.5


class Msg:
    def __init__(self, kind, **fields):
        self._kind = kind
        self.__dict__.update(fields)

    def get_type(self):
        return self._kind


def test_mavlink_state_gimbal_and_rangefinder():
    s = MavlinkState()
    assert s.sample("gimbal", 0.0) is None
    s.handle(Msg("GLOBAL_POSITION_INT", relative_alt=150_000, lat=469512340, lon=74412340))
    s.handle(Msg("ATTITUDE", roll=0.05, pitch=-0.1, yaw=1.2))
    sample = s.sample("gimbal", 0.0)
    assert sample.telemetry.altitude == 150.0 and sample.telemetry.pitch == 0.0
    assert sample.gnss == pytest.approx((46.951234, 7.441234))

    # gimbal pitched 30 degrees up from straight down
    half = math.radians(-60) / 2
    s.handle(Msg("GIMBAL_DEVICE_ATTITUDE_STATUS", q=[math.cos(half), 0, math.sin(half), 0]))
    assert math.isclose(math.degrees(s.sample("gimbal", 0.0).telemetry.pitch), 30.0, abs_tol=1e-6)

    s.handle(
        Msg(
            "DISTANCE_SENSOR",
            orientation=25,
            current_distance=14_200,
            min_distance=20,
            max_distance=40_000,
        )
    )
    assert s.sample("gimbal", 0.0).telemetry.altitude == 142.0


def test_mavlink_state_fixed_mount_follows_airframe():
    s = MavlinkState()
    s.handle(Msg("GLOBAL_POSITION_INT", relative_alt=100_000, lat=0, lon=0))
    s.handle(Msg("ATTITUDE", roll=0.1, pitch=-0.2, yaw=0.0))
    tel = s.sample("fixed", 0.0).telemetry
    assert math.isclose(tel.pitch, -0.2) and math.isclose(tel.roll, -0.1)


def test_mavlink_over_udp():
    pytest.importorskip("pymavlink")
    from pymavlink import mavutil

    from pharos.server.telemetry import MavlinkTelemetry

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    source = MavlinkTelemetry(f"udpin:127.0.0.1:{port}")
    sender = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}", source_system=1)
    try:
        deadline = time.monotonic() + 5
        sample = None
        while sample is None and time.monotonic() < deadline:
            pos = (469512340, 74412340, 600_000, 123_000)  # lat, lon, alt, relative alt (mm)
            sender.mav.global_position_int_send(0, *pos, 0, 0, 0, 0)
            sender.mav.attitude_send(0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0)
            time.sleep(0.05)
            sample = source.at(time.monotonic())
        assert sample is not None
        assert sample.telemetry.altitude == 123.0
        assert source.at(0.0).telemetry.altitude == 123.0  # before any sample: the oldest
    finally:
        source.close()
        sender.close()
