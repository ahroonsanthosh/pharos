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
