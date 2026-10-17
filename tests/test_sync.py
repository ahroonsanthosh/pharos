"""Video-to-log sync from the footage itself."""

import numpy as np
import pytest

from pharos.sim.flight import simulate
from pharos.sim.scenario import Scenario, TelemetrySpec
from pharos.sim.trajectory import TrajectorySpec
from pharos.sim.world import WorldSpec
from pharos.sync import estimate_sync
from pharos.telemetry import TelemetrySample
from pharos.types import Telemetry


@pytest.fixture(scope="module")
def wobbly_flight():
    """An airframe that rolls and pitches a few degrees, as fixed-wings do between turns."""
    sc = Scenario(
        seed=7,
        world=WorldSpec(size_m=700, villages=2, buildings_per_village=20),
        trajectory=TrajectorySpec(duration_s=30, margin_m=170, altitude=150, attitude_jitter_deg=4),
        telemetry=TelemetrySpec(attitude_noise_deg=0.0),
    )
    return simulate(sc)[1]


class DelayedTelemetry:
    """The simulated flight's telemetry, interpolated, ``delay_s`` later than its video."""

    def __init__(self, frames, delay_s):
        self.t = np.array([f.t for f in frames]) + delay_s
        tel = [f.telemetry for f in frames]
        self.values = np.array([[t.altitude, t.pitch, t.roll] for t in tel])
        self.yaw = np.unwrap([t.yaw for t in tel])

    def at(self, t):
        if not self.t[0] <= t <= self.t[-1]:
            return None
        alt, pitch, roll = (np.interp(t, self.t, self.values[:, k]) for k in range(3))
        return TelemetrySample(Telemetry(alt, float(np.interp(t, self.t, self.yaw)), pitch, roll))

    def close(self):
        pass


def test_finds_a_one_second_delay(wobbly_flight):
    rec = wobbly_flight
    telemetry = DelayedTelemetry(rec.frames, delay_s=1.0)
    result = estimate_sync(rec.frames, telemetry, rec.camera, span_s=1.5)
    assert result.offset_s == pytest.approx(1.0, abs=0.1)
    assert result.jitter_m < 0.5 * result.initial_jitter_m
