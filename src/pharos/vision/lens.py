"""Lens distortion: straighten frames from a real lens into the pinhole camera Pharos assumes.

Consumer drone cameras bend straight lines, most at the edges of the frame, where
a tilted or banked view looks furthest and matching relies on the outer ground.
Distortion follows OpenCV's model, coefficients ``(k1, k2, p1, p2, k3)``: radial
terms k1, k2, k3 and tangential terms p1, p2. Any shorter tuple leaves the rest
at zero; ``pharos lens`` measures them from views of a printed checkerboard.

The straightened view keeps the frame's size and is cropped just enough that no
pixel lies outside the original image: the matcher compares the whole frame
shape against the map, and black corners would look like ground.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import cv2
import numpy as np

from pharos.geometry import Camera


def coefficients(dist: Iterable[float]) -> np.ndarray:
    """OpenCV's five distortion coefficients from any shorter tuple."""
    d = [float(v) for v in dist]
    if len(d) > 5:
        raise ValueError("lens distortion takes at most 5 coefficients: k1 k2 p1 p2 k3")
    return np.array(d + [0.0] * (5 - len(d)))


class Lens:
    """Straightens frames of a distorting ``camera`` into the pinhole ``self.camera``."""

    def __init__(self, camera: Camera):
        self.source = camera
        w, h = camera.width, camera.height
        d = coefficients(camera.dist)
        K, _ = cv2.getOptimalNewCameraMatrix(
            camera.K, d, (w, h), 0.0, (w, h), centerPrincipalPoint=True
        )
        self.camera = Camera(w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        self._maps = cv2.initUndistortRectifyMap(camera.K, d, None, K, (w, h), cv2.CV_16SC2)

    def straighten(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if (w, h) != (self.source.width, self.source.height):
            size = (self.source.width, self.source.height)
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return cv2.remap(image, *self._maps, cv2.INTER_LINEAR)


def pinhole(camera: Camera) -> tuple[Camera, Lens | None]:
    """The camera the localizer works with, and the lens that gets frames there (if any)."""
    if not any(camera.dist):
        return replace(camera, dist=()), None
    lens = Lens(camera)
    return lens.camera, lens


def distort(image: np.ndarray, camera: Camera, margin: int) -> np.ndarray:
    """What a lens with ``camera.dist`` makes of ``image``, a pinhole view ``margin`` px wider.

    For the simulator: the pinhole view is rendered larger than the frame, since a
    barrel lens takes in more than its pinhole field of view at the edges.
    """
    xs, ys = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
    seen = np.stack([xs, ys], -1).reshape(-1, 1, 2).astype(np.float64)
    ideal = cv2.undistortPoints(seen, camera.K, coefficients(camera.dist), P=camera.K)
    ideal = ideal.reshape(camera.height, camera.width, 2).astype(np.float32) + margin
    return cv2.remap(image, ideal[..., 0], ideal[..., 1], cv2.INTER_LINEAR)


def margin_needed(camera: Camera) -> int:
    """Pixels beyond the frame that ``distort`` reads from, around the pinhole view."""
    w, h = camera.width - 1, camera.height - 1
    edge = np.array(
        [[x, y] for x in (0, w / 2, w) for y in (0, h / 2, h) if (x, y) != (w / 2, h / 2)],
        np.float64,
    ).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(edge, camera.K, coefficients(camera.dist), P=camera.K)
    ideal = ideal.reshape(-1, 2)
    over = max(-ideal.min(), (ideal[:, 0] - w).max(), (ideal[:, 1] - h).max(), 0.0)
    return int(np.ceil(over)) + 2


@dataclass(frozen=True)
class Calibration:
    camera: Camera  # with ``dist``
    hfov_deg: float  # horizontal field of view to pass with the coefficients (--hfov)
    rms_px: float  # how far, on average, the fitted model puts the checkerboard corners
    views: int  # images in which the board was found


def calibrate(images: Iterable[np.ndarray], board: tuple[int, int]) -> Calibration:
    """Field of view and distortion from views of a checkerboard with ``board`` inner corners.

    Take the views from different angles and distances, with the board reaching the
    edges and corners of the frame in some of them: that is where distortion shows.
    """
    cols, rows = board
    grid = np.zeros((cols * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    found, size = [], None
    for image in images:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            raise ValueError("all calibration images must have the same size")
        ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows))
        if ok:
            found.append(corners.astype(np.float32))
    if len(found) < 5:
        raise ValueError(
            f"found the {cols}x{rows} board in only {len(found)} images; need at least 5 "
            "(check --board counts inner corners, e.g. 9x6 for a 10x7-square board)"
        )
    rms, K, d, _, _ = cv2.calibrateCamera([grid] * len(found), found, size, None, None)
    w, h = size
    d = tuple(round(float(v), 6) for v in d.ravel()[:5])
    camera = Camera(w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), d)
    hfov = float(np.degrees(2 * np.arctan(w / (2 * camera.fx))))
    return Calibration(camera, hfov, float(rms), len(found))
