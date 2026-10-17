"""Sliding-window pose estimation: odometry, heading and map matches solved together.

Every processed frame is a pose (x, y, heading). Instead of folding each
measurement into a running estimate once and forgetting it, the window keeps
the last ``span_s`` seconds of frames and re-solves them jointly with robust
Gauss-Newton:

* **odometry** between consecutive frames with a heavy-tailed (Cauchy) error,
  divided by an odometry scale that is an unknown of every frame: lens
  distortion, field of view and height errors make odometry run long or short,
  and over uneven ground the height error changes along the track, most of all
  when the camera looks obliquely in a turn. The scale drifts as a random walk,
  so the uncertainty it leaves grows honestly where no match pins it;
* **gyro**: the heading change the autopilot reports, when there is one;
* **motion**: body-frame velocity changes slowly (coordinated flight), which
  carries the track across frames without odometry;
* **compass**: heading near the reported yaw plus a learned offset (loose);
* **map matches**: position and heading with each peak's own spread, as a
  max-mixture of "right" and "a look-alike somewhere around" (wide). A wrong
  match is outvoted by the odometry chain and the other matches of the
  window instead of being judged against a single prediction.

The same solver runs over a whole recorded flight (``span_s=inf``), so gaps
such as banked turns are bridged from both sides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from pharos.geometry import rot2, wrap_angle
from pharos.types import Fix, Telemetry
from pharos.vision.odometry import Motion


@dataclass(frozen=True)
class WindowConfig:
    span_s: float = 40.0  # frames older than this leave the window (as a prior)
    gyro_sigma_deg: float = 1.5  # error of the reported heading change between frames
    accel_sigma: float = 2.0  # m/s^2 of body-frame velocity change (manoeuvres, wind in turns)
    compass_sigma_deg: float = 20.0  # absolute heading around reported yaw + learned offset
    compass_walk_deg: float = 5.0  # how far the learned compass offset may move per window
    odometry_gain: float = 1.0  # times the odometry's own sigma
    odometry_robust: float = 3.0  # Cauchy scale of odometry residuals, in sigmas
    scale_sigma: float = 0.1  # prior on the odometry scale when a track starts
    # Odometry scale random walk, per sqrt(second), times (1 + scale_tilt * sin(tilt)):
    # terrain under an oblique view is further from the assumed height than under nadir.
    scale_walk: float = 0.01
    scale_tilt: float = 4.0
    match_sigma_m: float = 3.0
    match_heading_sigma_deg: float = 2.0
    outlier_sigma_m: float = 60.0  # a match on a look-alike lands somewhere around
    outlier_heading_sigma_deg: float = 60.0
    outlier_rate: float = 0.2  # share of matches expected on look-alikes
    # Errors shared by all matches (calibration residuals, attitude timing) do not average
    # out over many matches; reported uncertainties include this much in quadrature.
    systematic_sigma_m: float = 3.0
    marginal_inflation: float = 1.5  # dropped frames leave their information as a looser prior
    iterations: int = 8


@dataclass
class Node:
    t: float
    tel: Telemetry
    motion: Motion | None = None  # odometry from the previous node
    fix: Fix | None = None
    inlier: bool = False  # the fix's current classification
    pose: np.ndarray = field(default_factory=lambda: np.zeros(3))  # x, y, heading
    scale: float = 1.0  # odometry scale of the step into this frame


_V = 4  # unknowns per frame: x, y, heading, odometry scale


class PoseWindow:
    """Recent frames (pose and odometry scale each) plus the compass offset."""

    def __init__(
        self,
        cfg: WindowConfig,
        t: float,
        tel: Telemetry,
        pose: np.ndarray,
        cov: np.ndarray,
        compass_offset: float = 0.0,
        scale: float = 1.0,
    ):
        self.cfg = cfg
        self.nodes = [Node(t, tel, pose=np.asarray(pose, float).copy(), scale=scale)]
        self.compass_offset = compass_offset
        # Joint prior on (oldest x, y, heading, scale, compass offset): mean and whitener.
        prior_cov = np.zeros((5, 5))
        prior_cov[:3, :3] = cov
        prior_cov[3, 3] = cfg.scale_sigma**2
        prior_cov[4, 4] = math.radians(4 * cfg.compass_walk_deg) ** 2
        mean = np.concatenate([pose, [scale, compass_offset]])
        self.prior = (mean, _whitener(prior_cov))
        self.covs: list[np.ndarray] = [np.asarray(cov, float).copy()]  # marginal per node

    @property
    def scale(self) -> float:
        return self.nodes[-1].scale

    # -- building ---------------------------------------------------------

    def extend(
        self, t: float, tel: Telemetry, motion: Motion | None, guess: np.ndarray | None = None
    ) -> None:
        """Add a frame, initialized at ``guess`` or by dead reckoning from the last one."""
        last = self.nodes[-1]
        self.covs.append(self.covs[-1] + np.diag([25.0, 25.0, 0.01]))
        if guess is not None:
            node = Node(t, tel, motion, pose=np.asarray(guess, float).copy(), scale=last.scale)
            self.nodes.append(node)
            return
        pose = last.pose.copy()
        if motion is not None:
            pose[:2] += rot2(last.pose[2]) @ (last.scale * motion.d_body)
            pose[2] += motion.d_heading
        elif len(self.nodes) >= 2:
            prev = self.nodes[-2]
            dt0, dt1 = last.t - prev.t, t - last.t
            if dt0 > 0:
                body = rot2(prev.pose[2]).T @ (last.pose[:2] - prev.pose[:2]) * (dt1 / dt0)
                pose[:2] += rot2(last.pose[2]) @ body
        turn = _gyro(last.tel, tel)
        if turn is not None and motion is None:
            pose[2] += turn
        pose[2] = wrap_angle(pose[2])
        self.nodes.append(Node(t, tel, motion, pose=pose, scale=last.scale))

    def attach(self, fix: Fix) -> None:
        """A map match for the newest frame."""
        node = self.nodes[-1]
        node.fix = fix
        node.inlier = self._component(node.pose, fix)[0]

    def trim(self) -> None:
        """Drop frames older than the span; their information stays as a prior on the oldest."""
        cut = 0
        while cut < len(self.nodes) - 1 and self.nodes[-1].t - self.nodes[cut].t > self.cfg.span_s:
            cut += 1
        if not cut:
            return
        self.prior = self._marginalize(cut)
        self.nodes, self.covs = self.nodes[cut:], self.covs[cut:]
        self.nodes[0].motion = None  # its link to dropped frames is inside the prior now

    def _marginalize(self, cut: int) -> tuple[np.ndarray, np.ndarray]:
        """Prior on (frame ``cut``, compass offset) from the factors of frames before it.

        Schur complement of the dropped frames, over only the factors that leave the
        window with them, so evidence still in the window is not counted twice. A
        factor reaching past ``cut`` from a dropped frame is let go (the motion prior
        across the boundary): losing it is conservative.
        """
        n = len(self.nodes)
        x = self._state()
        r, J = self._linearize(x)
        J = J.tocsr()
        dropped = _V * cut
        kept = [dropped, dropped + 1, dropped + 2, dropped + 3, _V * n]
        rows = []
        for i in range(J.shape[0]):
            cols = J.indices[J.indptr[i] : J.indptr[i + 1]]
            if cols.min() < dropped and all(c < dropped or c in kept for c in cols):
                rows.append(i)
        Js = J[rows][:, list(range(dropped)) + kept].toarray()
        rs = r[rows]
        H, g = Js.T @ Js, Js.T @ rs
        Hdd, Hdm, Hmm = H[:dropped, :dropped], H[:dropped, dropped:], H[dropped:, dropped:]
        A = np.linalg.solve(Hdd + _DAMPING * np.eye(dropped), Hdm)
        info = Hmm - Hdm.T @ A
        grad = g[dropped:] - A.T @ g[:dropped]
        info = 0.5 * (info + info.T) + _DAMPING * np.eye(5)
        mean = x[kept] - np.linalg.solve(info, grad)
        mean[2], mean[4] = wrap_angle(mean[2]), wrap_angle(mean[4])
        walk = np.zeros((5, 5))  # the compass offset drifts a little over time
        walk[4, 4] = math.radians(self.cfg.compass_walk_deg) ** 2
        cov = np.linalg.inv(info) * self.cfg.marginal_inflation + walk
        return mean, _whitener(cov)

    # -- solving ----------------------------------------------------------

    def solve(self, covariances: str = "last") -> None:
        """Robust Gauss-Newton over the window; ``covariances``: "last", "all" or "none"."""
        n = len(self.nodes)
        x = self._state()
        for _ in range(self.cfg.iterations):
            r, J = self._linearize(x)
            H = (J.T @ J).tocsc()
            step = splu(H + _DAMPING * _eye(H.shape[0])).solve(-(J.T @ r))
            x += step
            x[2 : _V * n : _V] = wrap_angle(x[2 : _V * n : _V])
            if np.max(np.abs(step[: _V * n])) < 1e-3:
                break
        for k, nd in enumerate(self.nodes):
            nd.pose = x[_V * k : _V * k + 3].copy()
            nd.scale = float(x[_V * k + 3])
            if nd.fix is not None:
                nd.inlier = self._component(nd.pose, nd.fix)[0]
        self.compass_offset = float(x[-1])
        if covariances != "none":
            wanted = list(range(n)) if covariances == "all" else [n - 1]
            for k, cov in zip(wanted, self._marginals(wanted), strict=True):
                self.covs[k] = cov

    def _state(self) -> np.ndarray:
        frames = np.concatenate([[*nd.pose, nd.scale] for nd in self.nodes])
        return np.concatenate([frames, [self.compass_offset]])

    def _marginals(self, nodes: list[int]) -> list[np.ndarray]:
        """Marginal covariances (x, y, heading) of the given frames."""
        _, J = self._linearize(self._state())
        lu = splu((J.T @ J).tocsc() + _DAMPING * _eye(J.shape[1]))
        covs = []
        for k in nodes:
            cols = np.zeros((J.shape[1], 3))
            cols[_V * k : _V * k + 3] = np.eye(3)
            covs.append(lu.solve(cols)[_V * k : _V * k + 3])
        return covs

    # -- results ----------------------------------------------------------

    @property
    def pose(self) -> np.ndarray:
        return self.nodes[-1].pose

    def sigma(self, k: int = -1) -> float:
        """Position uncertainty of frame ``k`` (largest axis, systematic part included), metres."""
        solved = float(np.linalg.eigvalsh(self.covs[k][:2, :2]).max())
        return math.sqrt(solved + self.cfg.systematic_sigma_m**2)

    def heading_sigma(self, k: int = -1) -> float:
        return float(np.sqrt(self.covs[k][2, 2]))

    def support(self, separation_m: float = 40.0) -> int:
        """Independent inlier matches: runs of matches closer than ``separation_m`` count once."""
        count, last, flown = 0, -math.inf, 0.0
        for k, nd in enumerate(self.nodes):
            if k:
                flown += float(np.hypot(*(nd.pose[:2] - self.nodes[k - 1].pose[:2])))
            if nd.fix is not None and nd.inlier and flown - last >= separation_m:
                count, last = count + 1, flown
        return count

    def last_inlier_t(self) -> float | None:
        return next((nd.t for nd in reversed(self.nodes) if nd.fix is not None and nd.inlier), None)

    # -- internals --------------------------------------------------------

    def _component(self, pose: np.ndarray, fix: Fix) -> tuple[bool, np.ndarray, float]:
        """Max-mixture: (inlier?, position covariance, heading sigma) of the likelier component."""
        inlier, cov, sh = self._components(pose[None], [fix])
        return bool(inlier[0]), cov[0], float(sh[0])

    def _components(self, poses: np.ndarray, fixes: list[Fix]):
        """Vectorized ``_component`` for several (pose, fix) pairs."""
        cfg = self.cfg
        f = np.array([[fx.x, fx.y, fx.heading, *fx.spread] for fx in fixes])
        e = poses[:, :2] - f[:, :2]
        eh = wrap_angle(poses[:, 2] - f[:, 2])
        cov_in = np.empty((len(f), 2, 2))
        cov_in[:, 0, 0], cov_in[:, 0, 1], cov_in[:, 1, 1] = f[:, 3], f[:, 4], f[:, 5]
        cov_in[:, 1, 0] = f[:, 4]
        cov_in += cfg.match_sigma_m**2 * np.eye(2)
        sh_in = math.radians(cfg.match_heading_sigma_deg)
        sh_out = math.radians(cfg.outlier_heading_sigma_deg)
        maha = np.einsum("ka,ka->k", e, np.linalg.solve(cov_in, e[..., None])[..., 0])
        cost_in = 0.5 * (maha + (eh / sh_in) ** 2) + 0.5 * np.log(np.linalg.det(cov_in))
        cost_in += math.log(sh_in) - math.log(1.0 - cfg.outlier_rate)
        so = cfg.outlier_sigma_m
        cost_out = 0.5 * (np.sum(e**2, 1) / so**2 + (eh / sh_out) ** 2) + 2 * math.log(so)
        cost_out += math.log(sh_out) - math.log(cfg.outlier_rate)
        inlier = cost_in <= cost_out
        cov = np.where(inlier[:, None, None], cov_in, so**2 * np.eye(2))
        return inlier, cov, np.where(inlier, sh_in, sh_out)

    def _linearize(self, x: np.ndarray):
        """Whitened residuals and their sparse Jacobian at ``x`` (vectorized per factor kind)."""
        cfg, nodes = self.cfg, self.nodes
        n = len(nodes)
        ic = _V * n
        b = _Builder(_V * n + 1)
        F = x[: _V * n].reshape(n, _V)  # x, y, heading, scale per frame
        col = np.arange(n) * _V  # column of each frame's x
        t = np.array([nd.t for nd in nodes])
        yaw = np.array([np.nan if nd.tel.yaw is None else nd.tel.yaw for nd in nodes])
        tilt = np.array([nd.tel.tilt for nd in nodes])

        # prior on the oldest frame and the compass offset
        mean, W = self.prior
        e = x[[0, 1, 2, 3, ic]] - mean
        e[2], e[4] = wrap_angle(e[2]), wrap_angle(e[4])
        b.block(W @ e, np.array([0, 1, 2, 3, ic]), W)

        # odometry: true step / scale = measured step (the noise is in the measurement)
        k = np.array([i for i in range(1, n) if nodes[i].motion is not None], int)
        if len(k):
            m = [nodes[i].motion for i in k]
            d = np.array([mo.d_body for mo in m])
            s = cfg.odometry_gain * np.array([mo.sigma_m for mo in m])
            sh = cfg.odometry_gain * np.array([mo.sigma_heading for mo in m])
            dh = np.array([mo.d_heading for mo in m])
            Rt, dRt = _rt(F[k - 1, 2])
            delta = F[k, :2] - F[k - 1, :2]
            inv = 1.0 / F[k, 3]
            body = np.einsum("kab,kb->ka", Rt, delta)
            res = (inv[:, None] * body - d) / s[:, None]
            q = (1.0 / s)[:, None, None]
            J = np.concatenate(
                [
                    -inv[:, None, None] * Rt * q,
                    (inv[:, None] * np.einsum("kab,kb->ka", dRt, delta))[..., None] * q,
                    inv[:, None, None] * Rt * q,
                    (-(inv**2)[:, None] * body)[..., None] * q,
                ],
                axis=2,
            )
            j, i = col[k - 1], col[k]
            cols = np.stack([j, j + 1, j + 2, i, i + 1, i + 3], 1)
            b.block(res, cols, J, _cauchy(np.linalg.norm(res, axis=1), cfg.odometry_robust))
            res_h = wrap_angle(F[k, 2] - F[k - 1, 2] - dh) / sh
            Jh = np.stack([-1 / sh, 1 / sh], 1)[:, None, :]
            w = _cauchy(np.abs(res_h), cfg.odometry_robust)
            b.block(res_h[:, None], np.stack([j + 2, i + 2], 1), Jh, w)

        if n >= 2:
            # odometry scale: a random walk, looser when the camera looks obliquely
            k = np.arange(1, n)
            dt = np.maximum(t[k] - t[k - 1], 1e-3)
            oblique = np.sin(np.maximum(tilt[k], tilt[k - 1]))
            qs = cfg.scale_walk * np.sqrt(dt) * (1 + cfg.scale_tilt * oblique)
            res = (F[k, 3] - F[k - 1, 3]) / qs
            J = np.stack([-1 / qs, 1 / qs], 1)[:, None, :]
            b.block(res[:, None], np.stack([col[k - 1] + 3, col[k] + 3], 1), J)

        # gyro: the reported heading change between frames
        k = np.flatnonzero(np.isfinite(yaw[1:]) & np.isfinite(yaw[:-1])) + 1
        if len(k):
            sg = math.radians(cfg.gyro_sigma_deg)
            turn = wrap_angle(yaw[k] - yaw[k - 1])
            res = wrap_angle(F[k, 2] - F[k - 1, 2] - turn) / sg
            J = np.tile([-1 / sg, 1 / sg], (len(k), 1))[:, None, :]
            b.block(res[:, None], np.stack([col[k - 1] + 2, col[k] + 2], 1), J)

        # motion: body-frame velocity changes slowly
        if n >= 3:
            k = np.arange(2, n)
            dt1, dt2 = t[k - 1] - t[k - 2], t[k] - t[k - 1]
            ok = (dt1 > 0) & (dt2 > 0)
            k, dt1, dt2 = k[ok], dt1[ok], dt2[ok]
            if len(k):
                Ra, dRa = _rt(F[k - 2, 2])
                Rm, dRm = _rt(F[k - 1, 2])
                d1, d2 = F[k - 1, :2] - F[k - 2, :2], F[k, :2] - F[k - 1, :2]
                s = cfg.accel_sigma * 0.5 * (dt1 + dt2)
                v1 = np.einsum("kab,kb->ka", Ra, d1) / dt1[:, None]
                v2 = np.einsum("kab,kb->ka", Rm, d2) / dt2[:, None]
                res = (v2 - v1) / s[:, None]
                q1, q2 = (1 / (dt1 * s))[:, None, None], (1 / (dt2 * s))[:, None, None]
                J = np.concatenate(
                    [
                        Ra * q1,
                        -np.einsum("kab,kb->ka", dRa, d1)[..., None] * q1,
                        -Rm * q2 - Ra * q1,
                        np.einsum("kab,kb->ka", dRm, d2)[..., None] * q2,
                        Rm * q2,
                    ],
                    axis=2,
                )
                a, m_, c = col[k - 2], col[k - 1], col[k]
                cols = np.stack([a, a + 1, a + 2, m_, m_ + 1, m_ + 2, c, c + 1], 1)
                b.block(res, cols, J)

        # compass: heading near the reported yaw plus the learned offset
        k = np.flatnonzero(np.isfinite(yaw))
        if len(k):
            sc = math.radians(cfg.compass_sigma_deg)
            res = wrap_angle(F[k, 2] - yaw[k] - x[ic]) / sc
            J = np.tile([1 / sc, -1 / sc], (len(k), 1))[:, None, :]
            b.block(res[:, None], np.stack([col[k] + 2, np.full_like(k, ic)], 1), J)

        # map matches, each as the likelier of "right" and "a look-alike"
        k = np.array([i for i in range(n) if nodes[i].fix is not None], int)
        if len(k):
            fixes = [nodes[i].fix for i in k]
            _, cov, sh = self._components(F[k, :3], fixes)
            Wf = np.linalg.cholesky(np.linalg.inv(cov)).transpose(0, 2, 1)
            e = F[k, :2] - np.array([[f.x, f.y] for f in fixes])
            b.block(np.einsum("kab,kb->ka", Wf, e), np.stack([col[k], col[k] + 1], 1), Wf)
            eh = wrap_angle(F[k, 2] - np.array([f.heading for f in fixes])) / sh
            b.block(eh[:, None], (col[k] + 2)[:, None], (1 / sh)[:, None, None])
        return b.build()


# -- helpers -----------------------------------------------------------------

_DAMPING = 1e-9


def _eye(n: int):
    from scipy.sparse import identity

    return identity(n, format="csc")


def _rt(psi):
    """rot2(psi).T (world to body) and its derivative with respect to psi, stacked over psi."""
    c, s = np.cos(psi), np.sin(psi)
    Rt = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    dRt = np.stack([np.stack([-s, -c], -1), np.stack([c, -s], -1)], -2)
    return Rt, dRt


def _gyro(a: Telemetry, b: Telemetry) -> float | None:
    if a.yaw is None or b.yaw is None:
        return None
    return float(wrap_angle(b.yaw - a.yaw))


def _cauchy(d, c: float):
    return 1.0 / (1.0 + (np.asarray(d) / c) ** 2)


def _whitener(cov: np.ndarray) -> np.ndarray:
    """W with W.T @ W = inv(cov)."""
    L = np.linalg.cholesky(np.linalg.inv(cov))
    return L.T


class _Builder:
    """Collects whitened residual blocks and their Jacobians as sparse triplets."""

    def __init__(self, nvar: int):
        self.nvar = nvar
        self.r: list[np.ndarray] = []
        self.rows: list[np.ndarray] = []
        self.cols: list[np.ndarray] = []
        self.vals: list[np.ndarray] = []
        self.count = 0

    def block(self, res, cols, J, weight=None) -> None:
        """``res`` (K, r) or (r,), ``cols`` (K, c) or (c,), ``J`` (K, r, c) or (r, c)."""
        res, cols, J = np.asarray(res, float), np.asarray(cols), np.asarray(J, float)
        if res.ndim == 1:
            res, cols, J = res[None], cols[None], J[None]
        if weight is not None:
            sw = np.sqrt(np.atleast_1d(weight))
            res, J = res * sw[:, None], J * sw[:, None, None]
        K, r, c = J.shape
        rows = self.count + np.arange(K * r).reshape(K, r)
        self.count += K * r
        self.r.append(res.ravel())
        self.rows.append(np.broadcast_to(rows[:, :, None], (K, r, c)).ravel())
        self.cols.append(np.broadcast_to(cols[:, None, :], (K, r, c)).ravel())
        self.vals.append(J.ravel())

    def build(self):
        vals = np.concatenate(self.vals)
        keep = vals != 0.0
        rows, cols = np.concatenate(self.rows)[keep], np.concatenate(self.cols)[keep]
        J = csc_matrix((vals[keep], (rows, cols)), shape=(self.count, self.nvar))
        return np.concatenate(self.r), J
