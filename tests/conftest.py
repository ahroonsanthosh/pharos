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
