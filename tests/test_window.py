"""Sliding-window pose estimation on synthetic flights with look-alikes, ridges and turns."""

import math

import numpy as np
import pytest

from pharos.geometry import rot2, wrap_angle
from pharos.nav.window import PoseWindow, WindowConfig
from pharos.types import Fix, Telemetry
from pharos.vision.odometry import Motion

COMPASS_OFFSET = math.radians(7.0)
ODOMETRY_SHORT = 0.93  # odometry measures 7% too little


def racetrack(seconds=200.0, rate=2.0, speed=18.0, leg_s=40.0, turn_s=10.0):
    """Survey legs joined by 180-degree turns: (t, x, y, heading, turning) per frame."""
    out, x, y, h = [], 0.0, 0.0, 0.0
    dt, period = 1 / rate, leg_s + turn_s
    for k in range(int(seconds * rate)):
        t = k * dt
        turning = (t % period) >= leg_s
        out.append((t, x, y, h, turning))
        if turning:
            h += math.pi / turn_s * dt * (1 if (t // period) % 2 == 0 else -1)
        x += speed * dt * math.sin(h)
        y += speed * dt * math.cos(h)
    return out


def observations(track, rng, lookalike_every=7, ridge_every=3):
    """Telemetry, odometry and map matches as the localizer would see them."""
    obs = []
    lookalike = 0
    for k, (t, x, y, h, turning) in enumerate(track):
        tel = Telemetry(100.0, float(wrap_angle(h - COMPASS_OFFSET + rng.normal(0, 0.003))))
        motion = None
        if k:
            _, px, py, ph, _ = track[k - 1]
            body = rot2(ph).T @ np.array([x - px, y - py]) * ODOMETRY_SHORT
            motion = Motion(
                body + rng.normal(0, 1.5, 2),
                float(wrap_angle(h - ph)) + rng.normal(0, 0.015),
                2.0,
                0.02,
                200,
                1.0,
            )
        fix = None
        if not turning:
            spread = (0.0, 0.0, 0.0)
            if k % ridge_every == 0:  # a ridge along track: position known only across it
                u = np.array([math.sin(h), math.cos(h)])
                c = 400.0 * np.outer(u, u)
                spread = (c[0, 0], c[0, 1], c[1, 1])
            fx, fy = x + rng.normal(0, 3), y + rng.normal(0, 3)
            if k % (lookalike_every * 3) < 3:  # a run of three matches on a look-alike up-track
                lookalike += 1
                fx, fy = fx + 70 * math.sin(h), fy + 70 * math.cos(h)
            fix = Fix(fx, fy, float(h + rng.normal(0, 0.02)), 0.5, 0.2, spread=spread)
        obs.append((t, tel, motion, fix))
    return obs


def run_live(track, obs, cfg=None):
    cfg = cfg or WindowConfig()
    t0, x0, y0, h0, _ = track[0]
    w = PoseWindow(cfg, t0, obs[0][1], np.array([x0 + 3, y0 - 2, h0]), np.diag([25.0, 25.0, 0.01]))
    est = [(w.pose.copy(), w.sigma())]
    for t, tel, motion, fix in obs[1:]:
        w.extend(t, tel, motion)
        w.solve()
        if fix is not None:
            w.attach(fix)
            w.solve()
        w.trim()
        est.append((w.pose.copy(), w.sigma()))
    return w, est


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    track = racetrack(seconds=12)
    obs = observations(track, rng)
    cfg = WindowConfig(odometry_robust=1e9)
    w, _ = run_live(track, obs, cfg)
    x = w._state()
    x += rng.normal(0, 0.01, len(x))
    r0, J = w._linearize(x)
    J = J.toarray()
    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = 1e-6
        r1, _ = w._linearize(x + dx)
        np.testing.assert_allclose((r1 - r0) / 1e-6, J[:, i], atol=1e-3, rtol=1e-3)


def test_live_window_outvotes_lookalikes_and_learns_odometry_scale():
    rng = np.random.default_rng(1)
    track = racetrack()
    obs = observations(track, rng)
    w, est = run_live(track, obs)
    err = np.array(
        [np.hypot(p[0] - x, p[1] - y) for (p, _), (_, x, y, _, _) in zip(est, track, strict=True)]
    )
    sigma = np.array([s for _, s in est])
    assert np.median(err) < 4
    assert err.max() < 20  # never pulled onto the look-alike 70 m up-track
    assert np.mean(err <= 3 * sigma) > 0.95  # honest uncertainty, turns included
    assert w.scale == pytest.approx(1 / ODOMETRY_SHORT, abs=0.03)
    assert float(wrap_angle(w.compass_offset - COMPASS_OFFSET)) == pytest.approx(0, abs=0.02)


def test_whole_flight_solve_bridges_turns_from_both_sides():
    rng = np.random.default_rng(2)
    track = racetrack()
    obs = observations(track, rng)
    _, live = run_live(track, obs)
    t0, x0, y0, h0, _ = track[0]
    w = PoseWindow(
        WindowConfig(span_s=math.inf),
        t0,
        obs[0][1],
        np.array([x0, y0, h0]),
        np.diag([25.0, 25.0, 0.01]),
    )
    for t, tel, motion, fix in obs[1:]:
        w.extend(t, tel, motion)
        if fix is not None:
            w.attach(fix)
    w.solve(covariances="all")
    turning = np.array([tr[4] for tr in track])
    err_live = np.array(
        [np.hypot(p[0] - tr[1], p[1] - tr[2]) for (p, _), tr in zip(live, track, strict=True)]
    )
    err_all = np.array(
        [
            np.hypot(nd.pose[0] - tr[1], nd.pose[1] - tr[2])
            for nd, tr in zip(w.nodes, track, strict=True)
        ]
    )
    assert np.median(err_all[turning]) < np.median(err_live[turning])
    assert np.percentile(err_all, 95) < 8
    sigmas = np.array([w.sigma(k) for k in range(len(w.nodes))])
    assert np.mean(err_all <= 3 * sigmas) > 0.95


def test_dead_reckoning_stays_honest_without_matches():
    """Minutes of odometry alone, with gaps: the drift is covered by the reported sigma.

    Regression: information leaving the window was loosened once per dropped frame and
    lost at a gap in odometry, and the motion prior pulled an unpinned scale to zero.
    """
    rng = np.random.default_rng(3)
    track = racetrack(seconds=240)
    obs = observations(track, rng)
    obs = [
        (t, tel, None if k % 15 == 0 else motion, fix if t < 20 else None)
        for k, (t, tel, motion, fix) in enumerate(obs)
    ]
    w, est = run_live(track, obs)
    err = np.array(
        [np.hypot(p[0] - x, p[1] - y) for (p, _), (_, x, y, _, _) in zip(est, track, strict=True)]
    )
    sigma = np.array([s for _, s in est])
    assert np.all(np.isfinite(sigma))
    assert np.mean(err <= 3 * sigma) > 0.95
    assert sigma[-1] < 1000  # grows with distance, but stays useful
    assert 0.8 < w.scale < 1.3


def test_a_run_of_lookalike_matches_counts_as_one_place():
    """Sixteen consecutive matches on one look-alike 50 m up-track, sparse true matches.

    Overlapping frames share their errors. Counted as sixteen independent votes, the run
    outvoted odometry and the solve claimed 3 to 5 m while 50 m off.
    """
    rng = np.random.default_rng(5)
    track = racetrack(seconds=120)
    obs = []
    for k, (t, tel, motion, fix) in enumerate(observations(track, rng, 10**6, 10**6)):
        _, x, y, h, _ = track[k]
        if 120 <= k < 136:
            u = 50 * np.array([math.sin(h), math.cos(h)]) + rng.normal(0, 2, 2)
            fix = Fix(x + u[0], y + u[1], h, 0.5, 0.2)
        elif k % 12:
            fix = None
        obs.append((t, tel, motion, fix))
    t0, x0, y0, h0, _ = track[0]
    w = PoseWindow(
        WindowConfig(span_s=math.inf),
        t0,
        obs[0][1],
        np.array([x0, y0, h0]),
        np.diag([25.0] * 2 + [0.01]),
    )
    for t, tel, motion, fix in obs[1:]:
        w.extend(t, tel, motion)
        if fix is not None:
            w.attach(fix)
    w.solve(covariances="all")
    err = np.array(
        [
            np.hypot(nd.pose[0] - tr[1], nd.pose[1] - tr[2])
            for nd, tr in zip(w.nodes, track, strict=True)
        ]
    )
    sigma = np.array([w.sigma(k) for k in range(len(w.nodes))])
    assert err[110:146].max() < 25
    assert np.all(err <= 3 * sigma)
