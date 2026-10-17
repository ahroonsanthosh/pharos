"""Visual odometry: how far and which way the aircraft moved between two frames.

Consecutive frames come from the same camera minutes or seconds apart, so
ordinary feature matching works well between them (unlike frame-to-map
matching across seasons and sensors). Matched keypoints are projected onto
the ground with each frame's own altitude and camera tilt, and a rigid 2D
motion is fitted in metres. The result drives the localizer's motion model:
small search windows, heading changes without a compass, and coasting
through stretches where nothing matches the map.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pharos.geometry import Camera, ground_to_image, image_to_ground_points
from pharos.types import Frame


@dataclass(frozen=True)
class OdometryConfig:
    max_width: int = 800  # frames are downscaled to this for feature extraction
    features: int = 2000
    ratio: float = 0.75  # Lowe ratio test
    ransac_px: float = 3.0  # inlier threshold in frame pixels at the ground...
    ransac_relief: float = 0.02  # ...or this fraction of altitude (trees, roofs: parallax)
    min_inliers: int = 15
    max_range: float = 4.0  # ignore points further than this many altitudes away
    scale_tolerance: float = 0.2  # reject fits whose ground scale differs more than this
    # Error model. Attitude noise between the two frames shifts every projected point
    # alike, which a ground fit cannot tell from motion; altitude error scales distance.
    attitude_sigma_deg: float = 0.75  # the localizer scales this up if flights disagree
    relative_sigma: float = 0.1


@dataclass(frozen=True)
class Motion:
    """Motion of the aircraft from the previous frame to the current one.

    ``d_body`` is the new nadir point expressed in the previous frame's
    heading-up ground frame (right, forward), in metres. ``d_heading`` is the
    compass heading change (radians, clockwise positive).
    """

    d_body: np.ndarray
    d_heading: float
    sigma_m: float  # 1-sigma error of each component of d_body
    sigma_heading: float
    inliers: int
    scale: float  # ground-scale ratio between the frames (1 = altitudes agree)


class VisualOdometry:
    def __init__(self, camera: Camera, cfg: OdometryConfig | None = None):
        self.camera = camera
        self.cfg = cfg or OdometryConfig()
        self._sift = cv2.SIFT_create(nfeatures=self.cfg.features)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._prev: tuple[np.ndarray, np.ndarray, Frame, float] | None = None

    def reset(self) -> None:
        self._prev = None

    def step(self, frame: Frame, altitude_scale: float = 1.0) -> Motion | None:
        """Motion since the previous call, or None (first frame, too little overlap)."""
        uv, desc = self._features(frame.image)
        prev, self._prev = self._prev, (uv, desc, frame, altitude_scale)
        if prev is None or desc is None or prev[1] is None or len(uv) < self.cfg.min_inliers:
            return None
        uv_a, desc_a, frame_a, scale_a = prev
        pairs = self._matcher.knnMatch(desc, desc_a, k=2)
        good = [
            p[0] for p in pairs if len(p) == 2 and p[0].distance < self.cfg.ratio * p[1].distance
        ]
        if len(good) < self.cfg.min_inliers:
            return None
        ground_a = self._to_ground(uv_a[[m.trainIdx for m in good]], frame_a, scale_a)
        ground_b = self._to_ground(uv[[m.queryIdx for m in good]], frame, altitude_scale)
        ok = np.all(np.isfinite(ground_a), axis=1) & np.all(np.isfinite(ground_b), axis=1)
        if ok.sum() < self.cfg.min_inliers:
            return None
        return self._fit(ground_b[ok], ground_a[ok], frame.telemetry.altitude * altitude_scale)

    # -- internals --------------------------------------------------------

    def _features(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = min(1.0, self.cfg.max_width / gray.shape[1])
        if f < 1.0:
            gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        keypoints, desc = self._sift.detectAndCompute(gray, None)
        # back to full-resolution pixel coordinates (pixel centres)
        uv = np.array([k.pt for k in keypoints], np.float64).reshape(-1, 2)
        return (uv + 0.5) / f - 0.5, desc

    def _to_ground(self, uv: np.ndarray, frame: Frame, altitude_scale: float) -> np.ndarray:
        """Heading-up ground coordinates (right, forward) in metres, nadir at the origin."""
        tel = frame.telemetry
        altitude = tel.altitude * altitude_scale
        H = ground_to_image(self.camera, (0.0, 0.0, altitude), 0.0, tel.pitch, tel.roll)
        ground = image_to_ground_points(H, uv)
        far = np.hypot(ground[:, 0], ground[:, 1]) > self.cfg.max_range * altitude
        ground[far] = np.nan
        return ground

    def _fit(self, src: np.ndarray, dst: np.ndarray, altitude: float) -> Motion | None:
        cfg = self.cfg
        gsd = altitude / self.camera.fx
        M, inliers = cv2.estimateAffinePartial2D(
            src.astype(np.float32),
            dst.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=max(cfg.ransac_px * gsd, cfg.ransac_relief * altitude),
            maxIters=2000,
            confidence=0.995,
        )
        if M is None or inliers is None:
            return None
        n = int(inliers.sum())
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        if n < cfg.min_inliers or abs(scale - 1.0) > cfg.scale_tolerance:
            return None
        theta = float(np.arctan2(M[1, 0], M[0, 0]))  # counter-clockwise in (right, forward)
        R = M[:, :2] / scale
        t = M[:, 2]
        # Refit as a rigid motion: the scale belongs to altitude error, not to motion.
        mask = inliers.ravel().astype(bool)
        moved = src[mask] @ R.T + t
        residual = np.sqrt(np.mean(np.sum((moved - dst[mask]) ** 2, axis=1)))
        spread = np.sqrt(np.mean(np.sum((src[mask] - src[mask].mean(0)) ** 2, axis=1)))
        distance = float(np.hypot(*t))
        attitude = altitude * np.tan(np.radians(cfg.attitude_sigma_deg)) * np.sqrt(2)
        sigma_m = residual / np.sqrt(n) * 2 + cfg.relative_sigma * distance + attitude
        sigma_heading = residual / (np.sqrt(n) * max(spread, 1.0)) * 2 + np.radians(0.3)
        return Motion(t.copy(), -theta, float(sigma_m), float(sigma_heading), n, scale)
