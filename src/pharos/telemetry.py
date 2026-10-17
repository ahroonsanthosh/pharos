"""Telemetry that pairs with live or recorded footage: manual, DJI, MAVLink.

Every source answers "what did the aircraft report at time t?" with the
localizer's :class:`~pharos.types.Telemetry` (height above ground, camera
tilt, compass) plus, when the aircraft has it, its own GNSS position. The
GNSS position is never given to the localizer; it is only drawn next to the
visual estimate so every real flight doubles as an accuracy test.

Camera tilt follows :mod:`pharos.geometry`: ``pitch`` is the tilt of the
optical axis forward from straight down, ``roll`` to the right. A gimbal
reporting pitch -90 degrees (pointing straight down) is therefore tilt 0.
"""

from __future__ import annotations

import math
import re
import threading
import time
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from pharos.types import Telemetry


@dataclass(frozen=True)
class TelemetrySample:
    telemetry: Telemetry
    gnss: tuple[float, float] | None = None  # (lat, lon) reported by the aircraft


class TelemetrySource(Protocol):
    def at(self, t: float) -> TelemetrySample | None:
        """Telemetry at ``t``: video time for files, ``time.monotonic()`` for live feeds."""
        ...

    def close(self) -> None: ...


class ManualTelemetry:
    """Fixed values typed into the dashboard; editable while the feed runs."""

    def __init__(self, altitude: float, pitch: float = 0.0, roll: float = 0.0):
        self.telemetry = Telemetry(altitude=altitude, pitch=pitch, roll=roll)

    def update(self, **fields: float) -> None:
        self.telemetry = replace(self.telemetry, **fields)

    def at(self, t: float) -> TelemetrySample:
        return TelemetrySample(self.telemetry)

    def close(self) -> None:
        pass


# -- DJI ---------------------------------------------------------------------

_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->")
_SRT_NUMBER = re.compile(r"([A-Za-z_]+)\s*:\s*(-?\d+(?:\.\d+)?)")
_SRT_LEGACY_GPS = re.compile(r"GPS\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)")
_SRT_LEGACY_BARO = re.compile(r"BAROMETER\s*:\s*(-?[\d.]+)")


class SrtTelemetry:
    """Per-frame telemetry from the ``.SRT`` file DJI drones record next to each video.

    Handles the bracketed format of recent models (``[rel_alt: 120.5 ...]``,
    ``[gb_pitch: -90.0 ...]``) and the older ``GPS(lon,lat,alt) BAROMETER:h``
    format. Height above ground is taken as height above the takeoff point.
    """

    def __init__(self, path: str | Path):
        self.samples = parse_dji_srt(Path(path).read_text(errors="replace"))
        if not self.samples:
            raise ValueError(f"{path}: no telemetry found in SRT file")

    def at(self, t: float) -> TelemetrySample | None:
        # Subtitle entries are sorted by time; take the last one starting at or before t.
        lo, hi = 0, len(self.samples) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.samples[mid][0] <= t:
                lo = mid
            else:
                hi = mid - 1
        return self.samples[lo][1]

    def close(self) -> None:
        pass


def parse_dji_srt(text: str) -> list[tuple[float, TelemetrySample]]:
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        m = _SRT_TIME.search(block)
        if not m:
            continue
        h, mi, s, ms = m.groups()
        t = int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 10 ** len(ms)
        values = {k.lower(): float(v) for k, v in _SRT_NUMBER.findall(block[m.end() :])}

        altitude = values.get("rel_alt", values.get("altitude"))
        lat, lon = values.get("latitude"), values.get("longitude", values.get("longtitude"))
        if (gps := _SRT_LEGACY_GPS.search(block)) is not None:
            lon, lat = float(gps.group(1)), float(gps.group(2))
        if (baro := _SRT_LEGACY_BARO.search(block)) is not None:
            altitude = float(baro.group(1))
        if altitude is None:
            continue

        gimbal_pitch = values.get("gb_pitch", -90.0)
        yaw = values.get("gb_yaw")
        telemetry = Telemetry(
            altitude=altitude,
            yaw=None if yaw is None else math.radians(yaw),
            pitch=math.radians(90.0 + gimbal_pitch),
            roll=math.radians(values.get("gb_roll", 0.0)),
        )
        gnss = (lat, lon) if lat is not None and lon is not None and (lat or lon) else None
        out.append((t, TelemetrySample(telemetry, gnss)))
    out.sort(key=lambda item: item[0])
    return out


# -- MAVLink -----------------------------------------------------------------


class MavlinkTelemetry:
    """Live telemetry from ArduPilot / PX4 over MAVLink (needs ``pharos[server]``).

    Listens on ``endpoint`` (default ``udpin:0.0.0.0:14551``, a port ground
    stations forward to). Height comes from a downward rangefinder when one
    reports, else from ``GLOBAL_POSITION_INT.relative_alt``. Camera tilt comes
    from the gimbal (``GIMBAL_DEVICE_ATTITUDE_STATUS`` or ``MOUNT_STATUS``) or,
    with ``mount="fixed"``, from the airframe attitude plus ``mount_pitch``.
    """

    def __init__(
        self, endpoint: str = "udpin:0.0.0.0:14551", mount: str = "gimbal", mount_pitch: float = 0.0
    ):
        from pymavlink import mavutil  # optional dependency

        if mount not in ("gimbal", "fixed"):
            raise ValueError("mount must be 'gimbal' or 'fixed'")
        self.mount = mount
        self.mount_pitch = mount_pitch
        self.state = MavlinkState()
        self._history: deque[tuple[float, TelemetrySample]] = deque(maxlen=2000)
        self._conn = mavutil.mavlink_connection(endpoint, source_system=254)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mavlink", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            msg = self._conn.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue
            self.state.handle(msg)
            sample = self.state.sample(self.mount, self.mount_pitch)
            if sample is not None:
                self._history.append((time.monotonic(), sample))

    def at(self, t: float) -> TelemetrySample | None:
        """The last sample received at or before monotonic time ``t``."""
        history = list(self._history)
        if not history:
            return None
        i = bisect_right([when for when, _ in history], t)
        return history[max(i - 1, 0)][1]

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._conn.close()


class MavlinkState:
    """Latest values from the MAVLink messages pharos cares about."""

    STALE_S = 3.0

    def __init__(self):
        self.altitude: float | None = None
        self.range: tuple[float, float] | None = None  # (metres, receive time)
        self.attitude = (0.0, 0.0, 0.0)  # roll, pitch, yaw of the airframe (rad, NED)
        self.gimbal: tuple[float, float] | None = None  # roll, pitch (rad, earth frame)
        self.gnss: tuple[float, float] | None = None
        self.yaw: float | None = None

    def handle(self, msg) -> None:
        kind = msg.get_type()
        if kind == "GLOBAL_POSITION_INT":
            self.altitude = msg.relative_alt / 1000.0
            if msg.lat or msg.lon:
                self.gnss = (msg.lat / 1e7, msg.lon / 1e7)
        elif kind == "ATTITUDE":
            self.attitude = (msg.roll, msg.pitch, msg.yaw)
            self.yaw = msg.yaw
        elif kind == "DISTANCE_SENSOR" and msg.orientation == 25:  # MAV_SENSOR_ROTATION_PITCH_270
            if msg.min_distance <= msg.current_distance <= msg.max_distance:
                self.range = (msg.current_distance / 100.0, time.monotonic())
        elif kind == "GIMBAL_DEVICE_ATTITUDE_STATUS":
            w, x, y, z = msg.q
            roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
            self.gimbal = (roll, pitch)
        elif kind == "MOUNT_STATUS":
            self.gimbal = (math.radians(msg.pointing_b / 100), math.radians(msg.pointing_a / 100))

    def sample(self, mount: str, mount_pitch: float) -> TelemetrySample | None:
        altitude = self.altitude
        if self.range is not None and time.monotonic() - self.range[1] < self.STALE_S:
            altitude = self.range[0]
        if altitude is None:
            return None
        roll, pitch, _ = self.attitude
        if mount == "fixed":
            # A body-fixed downward camera tilts forward with the nose and left with the right wing.
            tilt, side = pitch + mount_pitch, -roll
        elif self.gimbal is not None:
            tilt, side = math.pi / 2 + self.gimbal[1], self.gimbal[0]
        else:
            tilt, side = 0.0, 0.0  # stabilized gimbal assumed to point straight down
        return TelemetrySample(Telemetry(altitude, self.yaw, tilt, side), self.gnss)
