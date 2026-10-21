"""Matcher and localizer against a small simulated flight."""

import numpy as np
import pytest

from pharos.eval.bench import localize
from pharos.eval.metrics import evaluate
from pharos.nav.localizer import Localizer
from pharos.types import Status
from pharos.vision.matcher import CorrelationMatcher, SearchRegion


def test_local_search_recovers_pose(small_sim):
    geomap, rec = small_sim
    matcher = CorrelationMatcher(geomap, rec.camera)
    for i in (0, 20, 40):
        frame, g = rec.frames[i], rec.truth[i]
        prior = SearchRegion((g.x + 12, g.y - 9), 30, g.heading + np.radians(5), np.radians(8))
        fix = matcher.match(frame, prior)[0]
        assert np.hypot(fix.x - g.x, fix.y - g.y) < 5
        assert abs(np.degrees(fix.heading - g.heading)) < 2


def test_local_search_with_tilted_camera(tilted_sim):
    geomap, rec = tilted_sim
    matcher = CorrelationMatcher(geomap, rec.camera)
    frame, g = rec.frames[4], rec.truth[4]
    prior = SearchRegion((g.x - 8, g.y + 10), 25, g.heading, np.radians(6))
    fix = matcher.match(frame, prior)[0]
    assert np.hypot(fix.x - g.x, fix.y - g.y) < 6


def test_global_search_with_unknown_heading(small_sim):
    geomap, rec = small_sim
    matcher = CorrelationMatcher(geomap, rec.camera)
    frame, g = rec.frames[10], rec.truth[10]
    best = matcher.match(frame, SearchRegion())[0]
    assert np.hypot(best.x - g.x, best.y - g.y) < 10


def test_global_search_turns_candidates_around_only_where_the_heading_allows(small_sim):
    geomap, rec = small_sim
    matcher = CorrelationMatcher(geomap, rec.camera)
    frame, g = rec.frames[10], rec.truth[10]

    def turned(region):
        fixes = matcher.match(frame, region)
        return [f for f in fixes if abs(np.angle(np.exp(1j * (f.heading - g.heading)))) > np.pi / 2]

    assert turned(SearchRegion((g.x, g.y), 450))  # no heading: both ways compete
    assert not turned(SearchRegion((g.x, g.y), 450, g.heading, np.radians(45)))  # a compass


def test_localizer_tracks_flight(small_sim):
    geomap, rec = small_sim
    estimates = localize(geomap, rec)
    m = evaluate(estimates, rec.truth)
    assert estimates[-1].status is Status.TRACKING
    assert m["availability"] > 0.5
    assert m["err_median_m"] < 6
    assert m["confident_wrong"] == 0


def test_track_survives_a_blackout_without_false_fixes(small_sim):
    """Six seconds of blank frames: coast, never claim a wrong position, recover quickly."""
    from pharos.types import Frame

    geomap, rec = small_sim
    blank = range(24, 36)
    frames = [
        Frame(f.t, f.image * 0 + 110, f.telemetry) if i in blank else f
        for i, f in enumerate(rec.frames)
    ]
    loc = Localizer(geomap, rec.camera)
    estimates = [loc.process(f) for f in frames]
    m = evaluate(estimates, rec.truth)
    assert m["confident_wrong"] == 0
    # dead reckoning is trusted only while its uncertainty is small, then it coasts
    assert all(e.status is not Status.TRACKING or e.sigma_xy <= 15 for e in estimates[24:36])
    assert any(e.status is Status.COASTING for e in estimates[24:36])
    recovered = [i for i in range(36, len(estimates)) if estimates[i].status is Status.TRACKING]
    # trusted again after two fresh matches 40 m of flight apart (12 m/s, 2 frames/s)
    assert recovered and recovered[0] <= 36 + 9


def test_frames_on_the_ground_or_facing_the_horizon_are_skipped(small_sim):
    """Takeoff in the video: no crash, no fixes, then normal tracking once airborne."""
    from dataclasses import replace

    from pharos.types import Frame

    geomap, rec = small_sim
    ground = [
        Frame(t, rec.frames[0].image, replace(rec.frames[0].telemetry, altitude=a))
        for t, a in ((-3.0, 0.0), (-2.5, 0.5))
    ]
    horizon = Frame(-2.0, rec.frames[0].image, replace(rec.frames[0].telemetry, pitch=1.3))
    loc = Localizer(geomap, rec.camera)
    early = [loc.process(f) for f in [*ground, horizon]]
    assert all(e.status is Status.LOST for e in early)
    estimates = [loc.process(f) for f in rec.frames]
    assert estimates[-1].status is Status.TRACKING


def test_odometry_implying_impossible_speed_is_ignored(small_sim):
    import numpy as np

    from pharos.vision.odometry import Motion

    geomap, rec = small_sim
    loc = Localizer(geomap, rec.camera)
    loc._last_t = 10.0
    step = Motion(np.array([0.0, 9.0]), 0.0, 3.0, 0.01, 100, 1.0)
    jump = Motion(np.array([0.0, 37.0]), 0.0, 6.8, 0.01, 100, 1.0)  # 74 m/s
    assert loc._plausible(step, 10.5) is step
    assert loc._plausible(jump, 10.5) is None


def test_whole_flight_solve_covers_at_least_the_live_track(small_sim):
    geomap, rec = small_sim
    loc = Localizer(geomap, rec.camera)
    live = [loc.process(f) for f in rec.frames]
    smoothed = loc.smooth()
    m_live, m_all = evaluate(live, rec.truth), evaluate(smoothed, rec.truth)
    assert m_all["coverage_25m"] >= m_live["coverage_25m"]
    assert m_all["misleading"] == 0
    assert m_all["err_median_m"] < 6


def test_lost_track_keeps_dead_reckoning_and_reacquires(small_sim):
    """A blackout beyond the coast limit: the position stays reported with an honest
    uncertainty, and a new track is confirmed soon after the frames come back."""
    from pharos.nav.localizer import LocalizerConfig
    from pharos.types import Frame

    geomap, rec = small_sim
    blank = range(24, 40)  # the track is confirmed at frame 20
    frames = [
        Frame(f.t, f.image * 0 + 110, f.telemetry) if i in blank else f
        for i, f in enumerate(rec.frames)
    ]
    loc = Localizer(geomap, rec.camera, LocalizerConfig(max_coast_s=3.0))
    estimates, reckoning = [], []
    for f in frames:
        estimates.append(loc.process(f))
        reckoning.append(loc.dead_reckoning is not None)
    assert any(reckoning[i] for i in blank)  # the track was lost...
    for e, g in zip(estimates[24:40], rec.truth[24:40], strict=True):
        assert e.status is not Status.LOST  # ...but never stopped reporting a position
        assert np.hypot(e.x - g.x, e.y - g.y) <= 3 * e.sigma_xy
    assert evaluate(estimates, rec.truth)["confident_wrong"] == 0
    recovered = [i for i in range(40, len(estimates)) if estimates[i].status is Status.TRACKING]
    assert recovered and recovered[0] <= 40 + 9


def test_untrusted_track_gives_way_to_consistent_rejected_matches(small_sim):
    """Matches the window rejects, all pointing to one place, end an untrusted track."""
    from pharos.nav.window import PoseWindow, WindowConfig
    from pharos.types import Fix, Telemetry

    geomap, rec = small_sim
    tel = Telemetry(170.0, 0.0)
    for offsets, expected in (
        ([(35, 0), (36, 2), (34, -1)], True),
        ([(35, 0), (-40, 10), (5, 45)], False),
    ):
        loc = Localizer(geomap, rec.camera)
        loc.window = PoseWindow(
            WindowConfig(), 0.0, tel, np.zeros(3), np.diag([400.0, 400.0, 0.01])
        )
        results = []
        for k, (dx, dy) in enumerate(offsets):
            results.append(loc._contradicted(float(k), Fix(dx, dy, 0.0, 0.4, 0.2, excess=5.0)))
        assert results[:2] == [False, False]
        assert results[2] is expected


def _misreported(frame, factor):
    """The frame with its height reported ``factor`` times too high, above takeoff."""
    from dataclasses import replace

    tel = replace(frame.telemetry, altitude=frame.telemetry.altitude * factor, above_takeoff=True)
    return replace(frame, telemetry=tel)


def test_matcher_takes_the_height_above_ground_from_the_terrain(small_sim):
    geomap, rec = small_sim
    matcher = CorrelationMatcher(geomap, rec.camera)
    i = 20
    frame, g = _misreported(rec.frames[i], 1.3), rec.truth[i]

    # The ground falls away eastwards 1 m in 10, so the true height applies only here.
    def height(xy):
        return g.altitude + 0.1 * (np.asarray(xy)[..., 0] - g.x)

    prior = SearchRegion((g.x + 12, g.y - 9), 30, g.heading, np.radians(8))
    off = matcher.match(frame, prior)[0]  # 30% off: nothing stands out
    fix = matcher.match(frame, prior, height=height)[0]
    assert np.hypot(fix.x - g.x, fix.y - g.y) < 5 and abs(fix.scale - 1) < 0.04
    assert fix.excess > 2 > off.excess

    best = matcher.match(frame, SearchRegion(), height=height)[0]
    assert np.hypot(best.x - g.x, best.y - g.y) < 10
    assert not matcher.match(frame, SearchRegion(), height=lambda xy: height(xy) - 1e4)


def test_heights_above_takeoff_become_heights_above_ground(hills_sim):
    """Over hills, with the takeoff point given or its elevation learned from the matches."""
    geomap, rec = hills_sim
    true = np.array([g.altitude for g in rec.truth])
    reported = np.array([f.telemetry.altitude for f in rec.frames])
    assert np.abs(reported - true).max() > 10
    elevation = float(geomap.terrain.at(np.array(rec.takeoff)))
    for takeoff in (rec.takeoff, None):
        loc = Localizer(geomap, rec.camera, takeoff=takeoff)
        estimates = [loc.process(f) for f in rec.frames]
        # learned from matches over relief, which measure heights to a few percent
        assert loc.takeoff_elevation == pytest.approx(elevation, abs=5.0)
        used = np.array([e["tel"].altitude for e in loc.log])
        corrected = np.array([not e["tel"].above_takeoff for e in loc.log])
        assert corrected.mean() > 0.6
        # the height above the ground the camera looks at, against the truth under it
        error = np.abs(used - true)[corrected]
        assert np.median(error) < 4 and error.max() < 9
        assert evaluate(estimates, rec.truth)["misleading"] == 0


def test_local_matches_over_steep_terrain_are_rectified_onto_it():
    from pharos.sim.flight import simulate
    from pharos.sim.scenario import Scenario
    from pharos.sim.trajectory import TrajectorySpec
    from pharos.sim.world import TerrainSpec, WorldSpec
    from pharos.vision.matcher import MatcherConfig

    sc = Scenario(
        seed=48,
        world=WorldSpec(size_m=700, villages=2, buildings_per_village=20),
        terrain=TerrainSpec(relief_m=150, wavelength_m=350),
        trajectory=TrajectorySpec(
            duration_s=6, margin_m=200, altitude=100, follow_terrain=True, tilt_deg=20
        ),
    )
    geomap, rec = simulate(sc)
    takeoff = float(geomap.terrain.at(np.array(rec.takeoff)))
    results = {}
    for relief in (False, True):
        matcher = CorrelationMatcher(geomap, rec.camera, MatcherConfig(terrain_relief=relief))
        rows = []
        for i in range(0, 12, 3):
            frame, g = rec.frames[i], rec.truth[i]
            height = geomap.terrain.height_above(takeoff + frame.telemetry.altitude)
            region = SearchRegion((g.x + 6, g.y - 4), 20, g.heading, np.radians(4))
            fix = matcher.match(frame, region, height=height)[0]
            rows.append((np.hypot(fix.x - g.x, fix.y - g.y), fix.excess))
        results[relief] = np.array(rows)
    assert results[True][:, 0].max() < 3
    assert np.median(results[True][:, 1]) > 1.5 * np.median(results[False][:, 1])


def test_coarse_features_are_the_fine_ones_reduced(small_sim):
    """Thin hedges and walls keep their direction at the coarse level (fine_features)."""
    from pharos.vision.matcher import MatcherConfig, _reduce

    geomap, rec = small_sim
    m = CorrelationMatcher(geomap, rec.camera)
    fine, gsd = m._features(0, False)
    coarse, coarse_gsd = m._features(1, True)
    assert coarse_gsd == 2 * gsd
    np.testing.assert_array_equal(coarse, _reduce(fine))
    blurred = CorrelationMatcher(geomap, rec.camera, MatcherConfig(fine_features=False))
    assert not np.allclose(blurred._features(1, True)[0], coarse)


def test_heights_compete_by_how_far_they_stand_out():
    from pharos.vision.matcher import _standardized

    rng = np.random.default_rng(0)
    low = rng.normal(0.4, 0.1, (50, 50)).astype(np.float32)  # small template: noisy, high
    high = rng.normal(0.2, 0.03, (50, 50)).astype(np.float32)  # large template: steady
    high[25, 25] = 0.35  # the true place, a clear outlier for its height
    everywhere = np.ones((50, 50), bool)
    assert low.max() > high.max()
    assert _standardized(high, everywhere).max() > _standardized(low, everywhere).max()
    assert np.isneginf(_standardized(low, ~everywhere)).all()


def test_candidates_move_by_odometry_at_their_own_height(small_sim):
    """Odometry runs at the lead's height above ground; a candidate over lower ground moved
    further, by the ratio of the heights."""
    from dataclasses import replace

    from pharos.terrain import Terrain
    from pharos.types import Fix, Telemetry
    from pharos.vision.odometry import Motion

    geomap, rec = small_sim
    x0, y0, _, y1 = geomap.extent
    ramp = np.array([[0.0, 100.0], [0.0, 100.0]], np.float32)  # rises 0.1 m per m eastwards
    geomap = replace(geomap, terrain=Terrain(ramp, 1000.0, x0, y1))
    loc = Localizer(geomap, rec.camera, takeoff=(x0, y0))
    above = Telemetry(150.0, 0.0, above_takeoff=True)
    for x, score in ((x0 + 100, 0.6), (x0 + 600, 0.4)):  # 140 and 90 m above the ground
        loc._candidate(Fix(x, y0 + 300, 0.0, score, 0.5), 0.0, False)
    loc._height = loc._height_field(above)
    frame = loc._above_ground(replace(rec.frames[0], telemetry=above), loc._belief(0.0))
    assert frame.telemetry.altitude == pytest.approx(140.0, abs=0.5)
    loc._motion, loc._last_t = Motion(np.array([0.0, 10.0]), 0.0, 0.5, 0.01, 50, 1.0), 0.0
    moved = []
    for h in loc.hypotheses:
        before = h.xy.copy()
        loc._predict(h, replace(frame, t=0.5))
        moved.append(float(np.hypot(*(h.xy - before))))
    assert moved == pytest.approx([10.0, 10.0 * 90 / 140], abs=0.1)


def test_a_flight_below_its_takeoff_point_is_still_flying(small_sim):
    """Taken off on high ground, an aircraft reads less than nothing over the valley it is
    50 m above; until it has left the ground, though, a low reading means on the ground."""
    from dataclasses import replace

    from pharos.nav.localizer import usable_view
    from pharos.terrain import Terrain
    from pharos.types import Telemetry

    geomap, rec = small_sim
    x0, y0, x1, y1 = geomap.extent
    ramp = np.array([[0.0, 100.0], [0.0, 100.0]], np.float32)  # rises 0.1 m per m eastwards
    geomap = replace(geomap, terrain=Terrain(ramp, 1000.0, x0, y1))
    loc = Localizer(geomap, rec.camera, takeoff=(x1, y0))  # about 70 m up the slope

    def seen(altitude):
        tel = Telemetry(altitude, 0.0, above_takeoff=True)
        loc._height = loc._height_field(tel)
        return loc._above_ground(replace(rec.frames[0], telemetry=tel), None).telemetry

    assert not usable_view(seen(0.5), loc.cfg)  # on the ground
    assert seen(40.0).altitude == 40.0  # climbed: a usable height above takeoff stays
    below = seen(-10.0)
    assert usable_view(below, loc.cfg) and not below.above_takeoff and below.altitude > 15


def test_a_known_start_is_tracked_from_the_first_frame(small_sim):
    """With the last fix before jamming, there is a position from the first frame and it
    is trusted sooner than from a cold start."""
    geomap, rec = small_sim
    g = rec.truth[0]
    known = localize(geomap, rec, start=((g.x + 8.0, g.y - 6.0), 10.0))
    cold = localize(geomap, rec)

    def first_trusted(estimates):
        return next(i for i, e in enumerate(estimates) if e.status is Status.TRACKING)

    assert all(e.status in (Status.TRACKING, Status.COASTING) for e in known)
    assert first_trusted(known) < first_trusted(cold)
    m = evaluate(known, rec.truth)
    assert m["misleading"] == 0 and m["coverage_25m"] > evaluate(cold, rec.truth)["coverage_25m"]


def test_a_known_start_survives_frames_without_odometry(small_sim):
    """Before odometry says how fast it flies, the track is bounded by how fast it can."""
    from pharos.nav.localizer import LocalizerConfig

    geomap, rec = small_sim
    g = rec.truth[0]
    loc = Localizer(geomap, rec.camera, LocalizerConfig(odometry=None))
    loc.start_at((g.x, g.y), 5.0)
    first = [loc.process(f) for f in rec.frames[:3]]
    assert all(e.status in (Status.TRACKING, Status.COASTING) for e in first)
    assert first[-1].sigma_xy < 50.0


def test_low_frames_match_on_a_finer_map_level(small_sim):
    """A frame from 40 m sees too little ground for a 1 m map; one from 170 m does not."""
    from pharos.geomap import GeoMap
    from pharos.vision.matcher import MatcherConfig

    _, rec = small_sim
    fine_map = GeoMap(np.zeros((400, 400, 3), np.uint8), 0.5)  # levels: 0.5, 1, 2 m
    m = CorrelationMatcher(fine_map, rec.camera, MatcherConfig(fine_level=1, coarse_level=2))
    frame = rec.frames[0]
    assert m._levels(frame, 170.0) == (1, 2)
    assert m._levels(frame, 40.0) == (0, 1)


def test_a_known_start_in_mid_air_learns_its_height_from_its_speed(hills_sim):
    """Heights read from a takeoff point 60 m below this ground (a video that begins in
    mid-air): the start's ground speed against odometry's step gives the true height."""
    from dataclasses import replace

    geomap, rec = hills_sim
    frames = [
        replace(f, telemetry=replace(f.telemetry, altitude=f.telemetry.altitude + 60.0))
        for f in rec.frames
    ]
    a, b = rec.truth[:2]
    velocity = ((b.x - a.x) / (b.t - a.t), (b.y - a.y) / (b.t - a.t))
    loc = Localizer(geomap, rec.camera)
    loc.start_at((a.x, a.y), 5.0, velocity=velocity)
    estimates = [loc.process(f) for f in frames]
    takeoff = float(geomap.terrain.at(np.array(rec.takeoff)))
    assert loc.takeoff_elevation == pytest.approx(takeoff - 60.0, abs=6.0)
    true = np.array([g.altitude for g in rec.truth])
    used = np.array([e["tel"].altitude for e in loc.log])
    assert np.median(np.abs(used - true) / true) < 0.05
    m = evaluate(estimates, rec.truth)
    assert m["misleading"] == 0 and m["coverage_25m"] > 0.7  # 0 without, 0.87 told the heights


def test_a_frozen_video_does_not_hold_the_track_back(small_sim):
    """A stalled link repeats its last picture for 3 s while the aircraft flies on: the
    repeats are not matched (they show where it was) and the track dead-reckons."""
    from dataclasses import replace

    geomap, rec = small_sim
    frames = list(rec.frames)
    for i in range(30, 36):
        frames[i] = replace(frames[i], image=frames[29].image)
    loc = Localizer(geomap, rec.camera)
    estimates = [loc.process(f) for f in frames]
    m = evaluate(estimates, rec.truth)
    assert m["misleading"] == 0 and m["within_3_sigma"] == 1.0
    assert all(loc.log[i]["fix"] is None and loc.log[i]["motion"] is None for i in range(30, 36))
