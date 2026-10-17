"""Kalman filter over planar position, velocity and heading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pharos.geometry import wrap_angle

# State: [x, y, vx, vy, heading]
_H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]], float)


@dataclass(frozen=True)
class FilterConfig:
    accel_sigma: float = 2.0  # m/s^2, white-noise acceleration
    heading_rate_sigma_deg: float = 20.0  # deg/s, heading random walk without a compass
    gyro_sigma_deg: float = 1.0  # deg/s, error on compass-derived heading changes
    init_velocity_sigma: float = 15.0  # m/s
    gate: float = 16.3  # chi-square, 3 dof, p = 0.999


class PoseFilter:
    def __init__(
        self,
        cfg: FilterConfig,
        x: float,
        y: float,
        heading: float,
        pos_sigma: float,
        heading_sigma: float,
    ):
        self.cfg = cfg
        self.x = np.array([x, y, 0.0, 0.0, heading])
        v = cfg.init_velocity_sigma**2
        self.P = np.diag([pos_sigma**2, pos_sigma**2, v, v, heading_sigma**2])

    @property
    def pos_sigma(self) -> float:
        """1-sigma horizontal error (root of the larger covariance eigenvalue)."""
        return float(np.sqrt(np.linalg.eigvalsh(self.P[:2, :2]).max()))

    @property
    def heading_sigma(self) -> float:
        return float(np.sqrt(self.P[4, 4]))

    def predict(self, dt: float, heading_change: float | None = None) -> None:
        """Constant-velocity step. ``heading_change`` comes from compass/gyro deltas if known."""
        F = np.eye(5)
        F[0, 2] = F[1, 3] = dt
        self.x = F @ self.x
        if heading_change is not None:
            self.x[4] += heading_change
            q_h = (np.radians(self.cfg.gyro_sigma_deg) ** 2) * dt
        else:
            q_h = (np.radians(self.cfg.heading_rate_sigma_deg) ** 2) * dt
        self.x[4] = wrap_angle(self.x[4])
        q = self.cfg.accel_sigma**2
        Q = np.zeros((5, 5))
        for i, j in ((0, 2), (1, 3)):
            Q[i, i] = q * dt**3 / 3
            Q[i, j] = Q[j, i] = q * dt**2 / 2
            Q[j, j] = q * dt
        Q[4, 4] = q_h
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        nu = z - _H @ self.x
        nu[2] = wrap_angle(nu[2])
        S = _H @ self.P @ _H.T + R
        return nu, S, float(nu @ np.linalg.solve(S, nu))

    def update(self, z: np.ndarray, R: np.ndarray) -> bool:
        """Fuse a measurement ``[x, y, heading]``. Returns False (no update) if gated out."""
        nu, S, d2 = self.innovation(z, R)
        if d2 > self.cfg.gate:
            return False
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[4] = wrap_angle(self.x[4])
        I_KH = np.eye(5) - K @ _H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form
        return True
