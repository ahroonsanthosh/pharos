"""Matcher and localizer against a small simulated flight."""

import numpy as np

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
    assert recovered and recovered[0] <= 36 + 6
