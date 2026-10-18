"""Coordinate conventions and camera geometry.

Every module in pharos uses these conventions; keep them in one place.

* World frame: local East-North-Up, metres. ``x`` east, ``y`` north, ``z`` up.
* Heading (yaw): radians, compass style. 0 = north, positive clockwise, so a
  drone flying east has heading pi/2.
* Camera attitude: a downward camera whose image "up" points along the
  heading. ``pitch`` tilts the optical axis forward from nadir and ``roll``
  tilts it to the right (both radians). Zero pitch and roll is a nadir view.
* Image pixels: ``(u, v)`` = (column, row) with integer values at pixel
  centres, matching OpenCV.
* Camera frame: OpenCV, x right, y down, z along the optical axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Camera axes (as world vectors) for a nadir view with heading 0:
# x_cam = east, y_cam = south (image down), z_cam = down.
_NADIR = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def wrap_angle(a):
    """Wrap an angle (radians, scalar or array) to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot2(heading: float) -> np.ndarray:
    """2D rotation taking heading-up (right, forward) coordinates into (east, north).

    A vector expressed as (right, forward) in the drone's frame maps to world
    (east, north) via ``rot2(heading) @ v``.
    """
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, s], [-s, c]])


def camera_rotation(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Rotation matrix whose columns are the camera axes expressed in world ENU."""
    return rot_z(-yaw) @ _NADIR @ rot_x(pitch) @ rot_y(roll)


# Autopilots describe attitude in aerospace frames: North-East-Down world, and
# body or gimbal axes x forward, y right, z down. A camera looking along x has
# image right along y and image down along z.
_NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_FORWARD_CAMERA = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def euler_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-NED rotation for aerospace (Z-Y-X) Euler angles in radians."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def camera_attitude(r_ned: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch and roll (as in :func:`camera_rotation`) of a camera looking
    along the x axis of a body or gimbal frame whose orientation is ``r_ned``.

    Exact for any attitude, so airframe and mount angles can be composed as
    rotations instead of added as angles.
    """
    return attitude_of(_NED_TO_ENU @ r_ned @ _FORWARD_CAMERA)


def attitude_of(R: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch and roll of a camera rotation in world ENU: the inverse of
    :func:`camera_rotation`."""
    yaw = np.arctan2(-R[0, 1], -R[1, 1])  # where image "up" points on the ground
    A = _NADIR.T @ rot_z(yaw) @ R  # = rot_x(pitch) @ rot_y(roll)
    return float(yaw), float(np.arctan2(A[2, 1], A[1, 1])), float(np.arctan2(A[0, 2], A[0, 0]))


def misaligned(
    yaw: float, pitch: float, roll: float, forward: float, right: float
) -> tuple[float, float, float]:
    """Attitude of a camera mounted ``forward``/``right`` (radians) off its reported attitude.

    The misalignment turns with the camera, so in a banked turn it is not simply
    added to pitch and roll.
    """
    return attitude_of(camera_rotation(yaw, pitch, roll) @ rot_x(forward) @ rot_y(right))


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics. Frames are assumed to be undistorted."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> Camera:
        f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2)
        return cls(width, height, f, f, (width - 1) / 2, (height - 1) / 2)

    @property
    def K(self) -> np.ndarray:  # noqa: N802 - standard name for the intrinsic matrix
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def scaled(self, factor: float) -> Camera:
        """Intrinsics of the same camera after resizing the image by ``factor``."""
        return Camera(
            round(self.width * factor),
            round(self.height * factor),
            self.fx * factor,
            self.fy * factor,
            (self.cx + 0.5) * factor - 0.5,
            (self.cy + 0.5) * factor - 0.5,
        )


def ground_to_image(
    camera: Camera,
    position: tuple[float, float, float],
    yaw: float,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> np.ndarray:
    """Homography mapping ground-plane points (x, y, 1) in metres to image pixels.

    ``position`` is the camera centre (x, y, height above the ground plane).
    """
    R = camera_rotation(yaw, pitch, roll)
    Rt = R.T
    t = -Rt @ np.asarray(position, dtype=float)
    return camera.K @ np.column_stack([Rt[:, 0], Rt[:, 1], t])


def image_to_ground_points(H_g2i: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Project image pixels (N, 2) onto the ground plane. Returns (N, 2) metres.

    Rays that do not hit the ground in front of the camera come back as NaN.
    """
    pts = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(H_g2i).T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pts[:, :2] / pts[:, 2:3]
    out[pts[:, 2] <= 1e-9] = np.nan
    return out


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) points."""
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]
