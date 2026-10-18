import numpy as np
import pytest

from pharos.sim.appearance import Look, render
from pharos.sim.trajectory import TrajectorySpec, generate_trajectory
from pharos.sim.world import ChangeSpec, Cover, WorldSpec, apply_changes, generate_world

SPEC = WorldSpec(size_m=300, villages=1, buildings_per_village=15, parcel_spacing_m=80)


def test_world_is_deterministic_and_varied():
    a, b = generate_world(SPEC, 7), generate_world(SPEC, 7)
    np.testing.assert_array_equal(a.cover, b.cover)
    assert a.shape == (600, 600)
    assert {Cover.CROP, Cover.ROAD} <= set(np.unique(a.cover).tolist())
    assert a.buildings


def test_changes_alter_the_world():
    w = generate_world(SPEC, 7)
    changed = apply_changes(w, ChangeSpec(replant=1.0, buildings_removed=1.0), 1)
    assert not changed.buildings
    assert np.sum(changed.cover == Cover.BUILDING) == 0
    assert np.any(changed.crop != w.crop)


def test_looks_render_differently():
    w = generate_world(SPEC, 7)
    summer = render(w, Look())
    snow = render(w, Look(season="snow"))
    assert summer.shape == (600, 600, 3) and summer.dtype == np.uint8
    assert snow.mean() > summer.mean() + 40
    with pytest.raises(ValueError):
        render(w, Look(season="monsoon"))


def test_trajectory_spacing_and_bounds():
    spec = TrajectorySpec(duration_s=40, fps=2, speed=10, margin_m=100)
    traj = generate_trajectory(spec, (0, 0, 800, 800), seed=3)
    xy = np.array([[s.x, s.y] for s in traj])
    assert len(traj) == 80
    np.testing.assert_allclose(np.linalg.norm(np.diff(xy, axis=0), axis=1), 5.0, rtol=0.05)
    assert xy.min() >= 100 - 1 and xy.max() <= 700 + 1


def test_grid_layout_is_rectilinear():
    spec = WorldSpec(
        size_m=400,
        layout="grid",
        parcel_spacing_m=100,
        major_roads=0,
        grid_road_every=2,
        villages=0,
        river=False,
        ponds=0,
    )
    w = generate_world(spec, 1)
    assert len(w.parcel_kind) >= 16
    assert np.sum(w.cover == Cover.ROAD) > 0  # lattice roads
    with pytest.raises(ValueError, match="layout"):
        generate_world(WorldSpec(size_m=200, layout="hexagons"), 1)


def test_sensor_errors_are_applied():
    from pharos.sim.flight import _SensorModel
    from pharos.sim.scenario import TelemetrySpec
    from pharos.types import GroundTruth

    truth = [GroundTruth(t * 0.5, 0.0, 0.0, 100.0 + t, 0.0) for t in range(10)]
    spec = TelemetrySpec(
        altitude_bias=0,
        altitude_noise_m=0,
        attitude_noise_deg=0,
        attitude_bias_deg=2.0,
        latency_s=1.0,
        altitude_offset_m=10,
    )
    tel = _SensorModel(spec, np.random.default_rng(0), truth).read(5)
    assert np.isclose(np.hypot(tel.pitch, tel.roll), np.radians(2.0))
    assert abs(tel.altitude - 103.0) == 10.0  # sampled 1 s (two frames) early, then offset
