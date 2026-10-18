"""Flight paths with ground-truth pose per frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pharos.geometry import wrap_angle
from pharos.sim.procedural import catmull_rom
from pharos.types import GroundTruth


@dataclass(frozen=True)
class TrajectorySpec:
    pattern: str = "random"  # "random" waypoints or "lawnmower" survey
    duration_s: float = 60.0
    fps: float = 2.0  # localization rate; real video is subsampled to this
    speed: float = 12.0  # m/s
    altitude: float = 180.0  # m above ground; over hills, above the takeoff point...
    follow_terrain: bool = False  # ...unless the aircraft keeps its height above ground
    altitude_variation: float = 10.0  # m, slow oscillation amplitude
    yaw_mode: str = "course"  # "course", "fixed", "spin"
    yaw_offset_deg: float = 0.0  # added to course (crabbing) or the fixed yaw
    spin_rate_deg_s: float = 6.0  # for yaw_mode "spin"
    tilt_deg: float = 0.0  # constant forward camera tilt
    attitude_jitter_deg: float = 1.0  # gimbal/airframe wobble
    margin_m: float = 250.0  # keep the path this far inside the map


def generate_trajectory(
    spec: TrajectorySpec, extent: tuple[float, float, float, float], seed: int
) -> list[GroundTruth]:
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = extent
    lo = np.array([x0 + spec.margin_m, y0 + spec.margin_m])
    hi = np.array([x1 - spec.margin_m, y1 - spec.margin_m])
    if np.any(hi <= lo):
        raise ValueError("map too small for the trajectory margin")
    length = spec.speed * spec.duration_s

    if spec.pattern == "random":
        ctrl = _random_waypoints(lo, hi, length, rng)
    elif spec.pattern == "lawnmower":
        ctrl = _lawnmower(lo, hi, rng)
    else:
        raise ValueError(f"unknown pattern {spec.pattern!r}")
    path = _resample(catmull_rom(ctrl, 30), spec.speed / spec.fps)

    n = int(spec.duration_s * spec.fps)
    if len(path) < n + 1:
        path = np.vstack([path, np.repeat(path[-1:], n + 1 - len(path), 0)])
    path = path[: n + 1]
    t = np.arange(n) / spec.fps

    step = np.diff(path, axis=0)
    course = np.arctan2(step[:, 0], step[:, 1])  # compass: atan2(east, north)
    offset = np.radians(spec.yaw_offset_deg)
    if spec.yaw_mode == "course":
        heading = course + offset
    elif spec.yaw_mode == "fixed":
        heading = np.full(n, rng.uniform(0, 2 * np.pi) + offset)
    elif spec.yaw_mode == "spin":
        heading = rng.uniform(0, 2 * np.pi) + np.radians(spec.spin_rate_deg_s) * t
    else:
        raise ValueError(f"unknown yaw_mode {spec.yaw_mode!r}")

    jitter = np.radians(spec.attitude_jitter_deg)
    wobble = [_smooth_noise(n, rng, spec.fps) * jitter for _ in range(3)]
    phase = rng.uniform(0, 2 * np.pi)
    altitude = spec.altitude + spec.altitude_variation * np.sin(2 * np.pi * t / 45.0 + phase)
    return [
        GroundTruth(
            t=float(t[i]),
            x=float(path[i, 0]),
            y=float(path[i, 1]),
            altitude=float(altitude[i]),
            heading=float(wrap_angle(heading[i] + wobble[0][i])),
            pitch=float(np.radians(spec.tilt_deg) + wobble[1][i]),
            roll=float(wobble[2][i]),
        )
        for i in range(n)
    ]


def _random_waypoints(lo, hi, length, rng) -> np.ndarray:
    pts = [rng.uniform(lo, hi)]
    heading = rng.uniform(0, 2 * np.pi)
    total = 0.0
    while total < length * 1.2:
        for _ in range(50):
            h = heading + rng.normal(0, 0.9)
            leg = rng.uniform(120, 350)
            p = pts[-1] + leg * np.array([np.sin(h), np.cos(h)])
            if np.all(p >= lo) and np.all(p <= hi):
                break
        else:
            h, leg = heading + np.pi, 120.0
            p = np.clip(pts[-1] + leg * np.array([np.sin(h), np.cos(h)]), lo, hi)
        pts.append(p)
        heading = h
        total += leg
    return np.array(pts)


def _lawnmower(lo, hi, rng) -> np.ndarray:
    spacing = 120.0
    xs = np.arange(lo[0], hi[0] + 1e-6, spacing)
    pts = []
    for i, x in enumerate(xs):
        ys = (lo[1], hi[1]) if i % 2 == 0 else (hi[1], lo[1])
        pts += [(x, ys[0]), (x, ys[1])]
    return np.array(pts)


def _resample(path: np.ndarray, step: float) -> np.ndarray:
    """Points spaced ``step`` metres apart along the polyline."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    q = np.arange(0.0, s[-1], step)
    return np.column_stack([np.interp(q, s, path[:, 0]), np.interp(q, s, path[:, 1])])


def _smooth_noise(n: int, rng, fps: float) -> np.ndarray:
    """Unit-variance noise correlated over about one second."""
    k = max(int(fps), 1)
    x = np.convolve(rng.standard_normal(n + 2 * k), np.ones(2 * k + 1), mode="same")[k : k + n]
    return x / (x.std() + 1e-9)
