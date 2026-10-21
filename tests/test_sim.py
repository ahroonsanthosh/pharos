from dataclasses import replace

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


def test_over_hills_heights_are_reported_above_the_takeoff_point():
    from pharos.sim.flight import _over_terrain
    from pharos.sim.trajectory import TrajectorySpec
    from pharos.sim.world import TerrainSpec, generate_terrain
    from pharos.terrain import Terrain
    from pharos.types import GroundTruth

    hills = generate_terrain(TerrainSpec(relief_m=50.0), 1000.0, seed=3)
    assert hills.elevation.min() == pytest.approx(0) and hills.elevation.max() == pytest.approx(50)
    assert generate_terrain(TerrainSpec(), 1000.0, seed=3) is None

    ramp = Terrain(np.array([[0.0, 40.0], [0.0, 40.0]], np.float32), 400.0, 0.0, 400.0)
    truth = [GroundTruth(0.0, x, 200.0, 100.0, 0.0) for x in (0.0, 200.0, 400.0)]
    held, above = _over_terrain(truth, ramp, TrajectorySpec())
    assert [g.altitude for g in held] == pytest.approx([100, 80, 60]) and above == [100] * 3
    kept, above = _over_terrain(truth, ramp, TrajectorySpec(follow_terrain=True))
    assert [g.altitude for g in kept] == pytest.approx([100] * 3)
    assert above == pytest.approx([100, 120, 140])
    with pytest.raises(ValueError, match="fly higher"):
        _over_terrain([replace(g, altitude=50.0) for g in truth], ramp, TrajectorySpec())


def test_irish_backdrops_bog_coast_and_flooding():
    from pharos.sim.world import Cover

    plain = generate_world(SPEC, 11)
    irish = generate_world(replace(SPEC, bog_fraction=0.4, coast=0.3), 11)
    land = irish.cover != Cover.WATER
    assert 0.25 < np.mean(irish.cover == Cover.WATER) < 0.4  # the sea, plus a few drains
    assert 0.25 < np.mean(irish.cover[land] == Cover.BOG) < 0.5
    assert np.mean(plain.cover == Cover.BOG) == 0
    flooded = apply_changes(plain, ChangeSpec(replant=0.0, flooded=0.5), 3)
    assert np.mean(flooded.cover == Cover.WATER) > np.mean(plain.cover == Cover.WATER) + 0.05


def test_overcast_light_casts_no_shadows():
    world = generate_world(SPEC, 7)
    sunny = render(world, Look(sun_elevation_deg=20)).astype(float)
    grey = render(world, Look(sun_elevation_deg=20, overcast=1.0)).astype(float)
    flat = render(world, Look(sun_elevation_deg=90, overcast=1.0)).astype(float)
    # with no direct light the sun's position no longer matters
    assert np.abs(grey - flat).mean() < 0.3 * np.abs(sunny - flat).mean()


def _roof_scene(roof_height):
    """A white 6 m square roof 15 m east of a camera 40 m up, over grey ground."""
    from pharos.geomap import GeoMap
    from pharos.geometry import Camera
    from pharos.sim.flight import render_frame
    from pharos.sim.scenario import CameraSpec
    from pharos.types import GroundTruth

    image = np.full((200, 200, 3), 90, np.uint8)
    image[94:106, 124:136] = 250  # x 12..18 m, y -3..3 m at 0.5 m per pixel, origin centred
    ground = GeoMap(image, 0.5, origin_x=-49.75, origin_y=49.75)
    heights = np.zeros((200, 200), np.float32)
    heights[94:106, 124:136] = roof_height
    camera = Camera.from_fov(320, 320, 80)
    spec = CameraSpec(blur=0, motion_blur_px=0, noise=0, vignetting=0, exposure_jitter=0)
    pose = GroundTruth(0.0, 0.0, 0.0, 40.0, 0.0)
    frame = render_frame(ground, camera, pose, spec, np.random.default_rng(0), heights=heights)
    v, u = np.nonzero(frame[..., 1] > 200)
    return np.hypot(u.mean() - camera.cx, v.mean() - camera.cy)


def test_tall_objects_lean_outwards_in_low_frames():
    """A roof 10 m up, seen from 40 m, sits 40/30 times further from the image centre."""
    flat, tall = _roof_scene(0.0), _roof_scene(10.0)
    assert tall / flat == pytest.approx(40 / 30, rel=0.05)


def test_fine_texture_shows_only_where_frames_resolve_it():
    from pharos.geomap import GeoMap
    from pharos.geometry import Camera
    from pharos.sim.flight import render_frame
    from pharos.sim.scenario import CameraSpec
    from pharos.types import GroundTruth

    ground = GeoMap(np.full((400, 400, 3), 120, np.uint8), 0.5, origin_x=-100, origin_y=100)
    camera = Camera.from_fov(320, 240, 80)
    spec = CameraSpec(blur=0, motion_blur_px=0, noise=0, vignetting=0, exposure_jitter=0)

    def detail(altitude, texture):
        pose = GroundTruth(0.0, 0.0, 0.0, altitude, 0.0)
        frame = render_frame(ground, camera, pose, spec, np.random.default_rng(0), texture=texture)
        return float(frame[60:180, 80:240, 1].std())

    assert detail(30.0, 0.1) > 4 * max(detail(30.0, 0.0), 0.5)  # 10 cm per pixel: grass shows
    assert detail(30.0, 0.1) > 2 * detail(150.0, 0.1)  # 50 cm per pixel: the image has it all


def test_stress_effects_happen_where_asked():
    """Frozen and lost video, stale telemetry, a pulled compass, a shifted and cut map."""
    from pharos.sim.flight import simulate
    from pharos.sim.scenario import CameraSpec, Scenario, TelemetrySpec

    base = Scenario(
        seed=5,
        world=WorldSpec(size_m=700, villages=1, buildings_per_village=10),
        trajectory=TrajectorySpec(duration_s=12, margin_m=170, altitude=150),
        camera=CameraSpec(width=320, height=240),
    )
    plain_map, plain = simulate(base)
    sc = replace(
        base,
        camera=replace(base.camera, freezes=((2.0, 2.0),), dropped=((6.0, 1.5),)),
        telemetry=TelemetrySpec(dropouts=((8.0, 2.0),), compass_jumps=((4.0, 2.0, 40.0),)),
        map_shift_m=(7.0, -3.0),
        map_crop_m=(0.0, 0.0, 0.0, 250.0),
    )
    geomap, rec = simulate(sc)
    t = [f.t for f in rec.frames]
    assert not any(6.0 <= x < 7.5 for x in t) and len(rec.truth) == len(rec.frames)
    frozen = [f for f in rec.frames if 2.0 <= f.t < 4.0]
    assert all(np.array_equal(f.image, frozen[0].image) for f in frozen)
    stale = [f.telemetry for f in rec.frames if 7.5 <= f.t < 10.0]
    assert all(s == stale[0] for s in stale)
    pulled = {f.t: f.telemetry.yaw for f in rec.frames}
    calm = {f.t: f.telemetry.yaw for f in plain.frames}
    jump = np.degrees(np.angle(np.exp(1j * (pulled[4.5] - calm[4.5]))))
    assert jump == pytest.approx(40.0, abs=5.0)
    assert geomap.origin_x - plain_map.origin_x == pytest.approx(7.0)
    assert plain_map.origin_y - geomap.origin_y == pytest.approx(253.0)
    assert plain_map.shape[0] - geomap.shape[0] == round(250.0 / geomap.gsd)
    assert any(not geomap.contains(g.x, g.y) for g in rec.truth)  # flown off the map


def test_tide_moves_the_shoreline():
    w = generate_world(replace(SPEC, coast=0.3, size_m=400), 3)
    sea = np.mean(w.cover == Cover.WATER)
    high = apply_changes(w, ChangeSpec(replant=0.0, tide_m=10.0), 1)
    low = apply_changes(w, ChangeSpec(replant=0.0, tide_m=-10.0), 1)
    assert np.mean(low.cover == Cover.WATER) < sea < np.mean(high.cover == Cover.WATER)
