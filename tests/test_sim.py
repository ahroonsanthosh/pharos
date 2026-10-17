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
