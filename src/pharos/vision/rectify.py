"""Orthorectification: warp a camera frame into a north-up ground patch.

Using the reported altitude and camera tilt, every frame is resampled onto the
ground plane at the map's ground sample distance. After this step the only
unknowns between the patch and the map are a 2D translation, the heading and
a small scale error (from altitude bias), which is what the matcher searches.

The patch is a disk centred on the ground point of the optical axis. Rotating
the heading hypothesis only rotates the content about that centre, so the disk
mask is identical for every heading and the correlator can share its
normalization across hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pharos.geometry import Camera, ground_to_image, image_to_ground_points, rot2


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


def disk_mask(radius_px: int) -> np.ndarray:
    n = 2 * radius_px + 1
    yy, xx = np.mgrid[:n, :n] - radius_px
    return (xx * xx + yy * yy) <= radius_px * radius_px


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
) -> np.ndarray:
    """North-up ground patch of size (2r+1)^2 centred on the optical-axis point.

    Patch pixel ``(u, v)`` is at ground offset ``rot2(heading) @ fp.center +
    ((u - r) * gsd, -(v - r) * gsd)`` (east, north) from the nadir point.
    """
    r = radius_px
    off = rot2(heading) @ fp.center
    A = np.array([[gsd, 0.0, off[0] - r * gsd], [0.0, -gsd, off[1] + r * gsd], [0.0, 0.0, 1.0]])
    H = ground_to_image(camera, (0.0, 0.0, altitude), heading, pitch, roll) @ A

    # Low-pass the frame first so downsampling onto the coarse patch grid does not alias.
    frame_gsd = np.hypot(altitude, np.linalg.norm(fp.center)) / camera.fx
    img = gray
    while gsd / frame_gsd >= 2.0 and min(img.shape) > 32:
        img = cv2.pyrDown(img)
        H = np.diag([0.5, 0.5, 1.0]) @ H
        frame_gsd *= 2.0
    ratio = gsd / frame_gsd
    if ratio > 1.0:
        img = cv2.GaussianBlur(img, (0, 0), 0.5 * ratio)
    n = 2 * r + 1
    return cv2.warpPerspective(
        img,
        H,
        (n, n),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def nadir_from_patch_center(center_xy: np.ndarray, heading: float, fp: Footprint) -> np.ndarray:
    """World position of the aircraft given the world position of the patch centre."""
    return np.asarray(center_xy) - rot2(heading) @ fp.center
