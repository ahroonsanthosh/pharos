"""Draw estimated and true tracks over the reference map."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from pharos.geomap import GeoMap
from pharos.types import Estimate, GroundTruth, Status

_TRUTH = (80, 200, 80)  # BGR
_COLORS = {Status.TRACKING: (255, 140, 0), Status.CONVERGING: (0, 165, 255)}


def draw_track(
    geomap: GeoMap,
    estimates: Sequence[Estimate],
    truth: Sequence[GroundTruth] | None = None,
    margin_m: float = 150.0,
) -> np.ndarray:
    """BGR image: map (dimmed), truth in green, estimates (blue tracking, orange converging)."""
    img = geomap.image if geomap.image.ndim == 3 else cv2.cvtColor(geomap.image, cv2.COLOR_GRAY2BGR)
    img = (img * 0.6).astype(np.uint8)
    pts_e = [
        (e, geomap.world_to_px(np.array([e.x, e.y]))) for e in estimates if e.status in _COLORS
    ]
    pts_t = geomap.world_to_px(np.array([[g.x, g.y] for g in truth])) if truth else np.empty((0, 2))

    if len(pts_t) > 1:
        cv2.polylines(img, [np.round(pts_t).astype(np.int32)], False, _TRUTH, 2, cv2.LINE_AA)
    for e, p in pts_e:
        cv2.circle(img, tuple(np.round(p).astype(int)), 3, _COLORS[e.status], -1, cv2.LINE_AA)

    all_pts = np.vstack([pts_t, *[p[None] for _, p in pts_e]]) if (len(pts_t) or pts_e) else None
    if all_pts is None:
        return img
    m = margin_m / geomap.gsd
    h, w = img.shape[:2]
    u0, v0 = np.clip(np.floor(all_pts.min(0) - m).astype(int), 0, [w, h])
    u1, v1 = np.clip(np.ceil(all_pts.max(0) + m).astype(int), 0, [w, h])
    return img[v0:v1, u0:u1]
