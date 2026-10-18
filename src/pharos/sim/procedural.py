"""Procedural noise and curves."""

from __future__ import annotations

import cv2
import numpy as np


def fbm(
    shape: tuple[int, int],
    rng: np.random.Generator,
    period_px: float,
    octaves: int = 4,
    persistence: float = 0.5,
) -> np.ndarray:
    """Fractal value noise, zero mean and unit variance. ``period_px`` sets the largest feature."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, period = 1.0, float(period_px)
    for _ in range(octaves):
        if period < 2:
            break
        gh, gw = int(h / period) + 4, int(w / period) + 4
        grid = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(grid, (int(gw * period), int(gh * period)), interpolation=cv2.INTER_CUBIC)
        oy, ox = rng.integers(0, int(period) + 1, 2)
        out += amp * up[oy : oy + h, ox : ox + w]
        amp *= persistence
        period /= 2
    out -= out.mean()
    return out / (out.std() + 1e-6)


def catmull_rom(points: np.ndarray, samples: int = 20) -> np.ndarray:
    """Smooth curve through ``points`` (N, 2), ``samples`` points per segment."""
    p = np.vstack([points[:1], points, points[-1:]])
    t = np.linspace(0, 1, samples, endpoint=False)[:, None]
    out = [
        0.5
        * (
            2 * p[i]
            + (p[i + 1] - p[i - 1]) * t
            + (2 * p[i - 1] - 5 * p[i] + 4 * p[i + 1] - p[i + 2]) * t**2
            + (-p[i - 1] + 3 * p[i] - 3 * p[i + 1] + p[i + 2]) * t**3
        )
        for i in range(1, len(p) - 2)
    ]
    return np.vstack([*out, points[-1:]])
