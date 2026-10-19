"""Orthorectification: warp a camera frame into a north-up ground patch.

Using the reported altitude and camera tilt, every frame is resampled onto the
ground plane at the map's ground sample distance. After this step the only
unknowns between the patch and the map are a 2D translation, the heading and
a small scale error (from altitude bias), which is what the matcher searches.

Over uneven ground the plane is the one under the patch centre, and ``relief``
gives the ground's height above it everywhere else: a valley side rising
towards the camera looks larger and shifts outwards, and projecting through
the terrain undoes that.

The patch is centred on the ground point of the optical axis, so a heading
hypothesis only turns its content about that centre. The matched template is
either all the ground the image covers (:func:`frame_mask`, which turns with
the heading) or the largest disk inside it, which is the same for every
heading.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from pharos.geometry import (
    Camera,
    camera_rotation,
    ground_to_image,
    image_to_ground_points,
    rot2,
)


@dataclass(frozen=True)
class Footprint:
    """Heading-independent geometry of one frame's ground coverage.

    ``center`` is the optical-axis ground point relative to the nadir point,
    in the heading-up frame (right, forward), metres. ``radius`` is the
    largest disk around it fully covered by the image; ``outline`` is the
    image border on the ground, relative to ``center`` in the same frame.
    """

    center: np.ndarray
    radius: float
    outline: np.ndarray

    @property
    def extent(self) -> float:
        """Distance from the centre to the furthest covered ground."""
        return float(np.max(np.linalg.norm(self.outline, axis=1)))


def footprint(camera: Camera, altitude: float, pitch: float, roll: float) -> Footprint:
    H = ground_to_image(camera, (0.0, 0.0, altitude), 0.0, pitch, roll)
    center = image_to_ground_points(H, np.array([[camera.cx, camera.cy]]))[0]
    if not np.all(np.isfinite(center)):
        raise ValueError("optical axis does not intersect the ground; camera tilt too large")
    n = 64
    w, h = camera.width - 1, camera.height - 1
    s = np.linspace(0, 1, n)
    border = np.concatenate(
        [
            np.column_stack([s * w, np.zeros(n)]),
            np.column_stack([np.full(n, w), s * h]),
            np.column_stack([s * w, np.full(n, h)]),
            np.column_stack([np.zeros(n), s * h]),
        ]
    )
    pts = image_to_ground_points(H, border)
    d = np.linalg.norm(pts - center, axis=1)
    radius = float(np.nanmin(d)) if np.any(np.isfinite(d)) else 0.0
    outline = pts[np.all(np.isfinite(pts), axis=1)] - center
    return Footprint(center=center, radius=radius, outline=outline)


@lru_cache(maxsize=32)
def disk_mask(radius_px: int) -> np.ndarray:
    """Pixels within ``radius_px`` of the centre of a (2r+1)^2 patch (shared: do not modify)."""
    n = 2 * radius_px + 1
    yy, xx = np.mgrid[:n, :n] - radius_px
    mask = (xx * xx + yy * yy) <= radius_px * radius_px
    mask.flags.writeable = False
    return mask


def frame_mask(fp: Footprint, heading: float, gsd: float, radius_px: int) -> np.ndarray:
    """The ground the whole image covers, in a north-up patch as :func:`rectify` renders it,
    a few pixels inside the border."""
    n = 2 * radius_px + 1
    east, north = (rot2(heading) @ fp.outline.T) / gsd
    pts = np.column_stack([radius_px + east, radius_px - north]).astype(np.float32)
    mask = np.zeros((n, n), np.uint8)
    # a frame seen on a plane is convex, whatever order its border was sampled in
    cv2.fillConvexPoly(mask, np.round(cv2.convexHull(pts)).astype(np.int32), 1)
    return cv2.erode(mask, np.ones((5, 5), np.uint8)).astype(bool) & disk_mask(radius_px)


def rectify(
    gray: np.ndarray,
    camera: Camera,
    altitude: float,
    pitch: float,
    roll: float,
    heading: float,
    gsd: float,
    radius_px: int,
    fp: Footprint,
    relief: np.ndarray | None = None,
) -> np.ndarray:
    """North-up ground patch of size (2r+1)^2 centred on the optical-axis point.

    Patch pixel ``(u, v)`` is at ground offset ``rot2(heading) @ fp.center +
    ((u - r) * gsd, -(v - r) * gsd)`` (east, north) from the nadir point.
    ``relief`` is the ground's height (m) above the plane ``altitude`` below the
    camera at each patch pixel; patch pixels the frame does not show repeat its
    edge (see :func:`patch_in_frame`).
    """
    r = radius_px
    off = rot2(heading) @ fp.center
    A = np.array([[gsd, 0.0, off[0] - r * gsd], [0.0, -gsd, off[1] + r * gsd], [0.0, 0.0, 1.0]])
    H = ground_to_image(camera, (0.0, 0.0, altitude), heading, pitch, roll) @ A

    # Low-pass the frame first so downsampling onto the coarse patch grid does not alias.
    frame_gsd = np.hypot(altitude, np.linalg.norm(fp.center)) / camera.fx
    img, shrink = gray, 1.0
    while gsd / frame_gsd >= 2.0 and min(img.shape) > 32:
        img = cv2.pyrDown(img)
        H = np.diag([0.5, 0.5, 1.0]) @ H
        shrink *= 0.5
        frame_gsd *= 2.0
    ratio = gsd / frame_gsd
    if ratio > 1.0:
        img = cv2.GaussianBlur(img, (0, 0), 0.5 * ratio)
    n = 2 * r + 1
    if relief is not None:
        uv = frame_pixels(camera, altitude, pitch, roll, heading, gsd, radius_px, fp, relief)
        maps = (uv * shrink).astype(np.float32)
        return cv2.remap(
            img, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
    return cv2.warpPerspective(
        img,
        H,
        (n, n),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def frame_pixels(
    camera: Camera,
    altitude: float,
    pitch: float,
    roll: float,
    heading: float,
    gsd: float,
    radius_px: int,
    fp: Footprint,
    relief: np.ndarray,
) -> np.ndarray:
    """Frame pixel ``(u, v)`` showing each pixel of the patch :func:`rectify` renders.

    The ground at a patch pixel is ``relief`` metres above the plane ``altitude``
    below the camera. Ground behind the camera comes back as -1.
    """
    r, n = radius_px, 2 * radius_px + 1
    off = rot2(heading) @ fp.center
    k = (np.arange(n) - r) * gsd
    ground = np.empty((n, n, 3))
    ground[..., 0] = off[0] + k[None, :]
    ground[..., 1] = off[1] - k[:, None]
    ground[..., 2] = np.asarray(relief) - altitude  # relative to the camera
    p = ground @ (camera.K @ camera_rotation(heading, pitch, roll).T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = p[..., :2] / p[..., 2:]
    uv[p[..., 2] <= 1e-6] = -1.0
    return uv


def patch_in_frame(uv: np.ndarray, camera: Camera) -> np.ndarray:
    """Which patch pixels the frame shows, given their :func:`frame_pixels`."""
    u, v = uv[..., 0], uv[..., 1]
    return (u >= 0) & (u <= camera.width - 1) & (v >= 0) & (v <= camera.height - 1)


def nadir_from_patch_center(center_xy: np.ndarray, heading: float, fp: Footprint) -> np.ndarray:
    """World position of the aircraft given the world position of the patch centre."""
    return np.asarray(center_xy) - rot2(heading) @ fp.center
