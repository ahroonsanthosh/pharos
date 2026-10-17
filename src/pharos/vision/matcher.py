"""Frame-to-map matchers: turn one frame into absolute position hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Protocol

import cv2
import numpy as np

from pharos.geomap import GeoMap
from pharos.geometry import Camera, rot2, wrap_angle
from pharos.types import Fix, Frame
from pharos.vision.correlate import Correlator, find_peaks, parabola_offset, subpixel
from pharos.vision.features import FeatureConfig, orientation_field
from pharos.vision.rectify import Footprint, disk_mask, footprint, nadir_from_patch_center, rectify


@dataclass(frozen=True)
class SearchRegion:
    """Where to look. ``center=None`` searches the whole map; ``heading=None`` all headings."""

    center: tuple[float, float] | None = None
    radius: float = float("inf")
    heading: float | None = None
    heading_span: float = np.pi  # half-width, radians


class Matcher(Protocol):
    def match(self, frame: Frame, region: SearchRegion, scale: float = 1.0) -> list[Fix]:
        """Best-first absolute fixes for ``frame`` inside ``region``.

        ``scale`` multiplies the reported altitude (the caller's running
        estimate of the barometric bias).
        """
        ...


@dataclass(frozen=True)
class MatcherConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    fine_level: int = 0  # map pyramid level for tracking and refinement
    coarse_level: int = 1  # map pyramid level for the global search
    max_radius_m: float = 160.0  # cap on the matched disk radius
    global_radius_m: float = 400.0  # regions wider than this use the coarse global search
    coarse_heading_step_deg: float = 6.0
    fine_heading_step_deg: float = 2.0
    scales: tuple[float, ...] = (0.94, 1.0, 1.06)  # altitude-correction hypotheses
    candidates: int = 5  # coarse candidates refined in a global search
    peak_separation_m: float = 40.0  # peaks closer than this are the same place
    sidelobe_m: float = 8.0  # tracking: rival peaks must be at least this far from the best
    min_overlap: float = 0.75


@dataclass
class _Search:
    """Result of scoring every (heading, scale) hypothesis over one ROI."""

    best: np.ndarray  # max score over hypotheses, per template offset
    stack: np.ndarray | None  # (n_headings, n_scales, H, W) scores if kept
    best_idx: np.ndarray  # flat hypothesis index of ``best``
    headings: np.ndarray
    scales: tuple[float, ...]
    footprints: dict[float, Footprint]
    origin: tuple[int, int]  # level-pixel (u, v) of ROI pixel (0, 0)
    radius_px: int
    level: int


class CorrelationMatcher:
    """Dense orientation-field correlation against the map (CPU, no learned weights).

    * Local mode (tracking): search a small window around a predicted pose at
      the fine level over a few headings and altitude-scale hypotheses, then
      refine position, heading and scale to sub-sample precision.
    * Global mode (lost / first fix): correlate over the whole region at the
      coarse level for every heading, then refine the best few candidates
      with a local search.
    """

    def __init__(self, geomap: GeoMap, camera: Camera, cfg: MatcherConfig | None = None):
        self.map = geomap
        self.camera = camera
        self.cfg = cfg or MatcherConfig()
        self._features = cache(self._compute_features)

    def _compute_features(self, level: int) -> tuple[np.ndarray, float]:
        img, gsd = self.map.level(level)
        return orientation_field(img, None, self.cfg.features), gsd

    def match(self, frame: Frame, region: SearchRegion, scale: float = 1.0) -> list[Fix]:
        gray = _gray(frame.image)
        if region.center is None or region.radius > self.cfg.global_radius_m:
            return self._global(gray, frame, region, scale)
        fix = self._local(gray, frame, region, scale)
        return [] if fix is None else [fix]

    # -- search modes -----------------------------------------------------

    def _global(self, gray, frame, region, scale) -> list[Fix]:
        cfg = self.cfg
        step = np.radians(cfg.coarse_heading_step_deg)
        res = self._search(
            gray,
            frame,
            cfg.coarse_level,
            region.center,
            region.radius,
            _heading_grid(region.heading, region.heading_span, step),
            (scale,),
            keep=False,
        )
        if res is None:
            return []
        _, gsd = self._features(cfg.coarse_level)
        sep = max(int(cfg.peak_separation_m / gsd), 1)
        fixes = []
        for r, c, _ in find_peaks(res.best, cfg.candidates, sep):
            i = int(res.best_idx[r, c])
            heading = float(res.headings[i])
            x, y = self._to_world(res, r, c, heading, res.footprints[scale])
            # Line-dominated scenes can match almost as well turned around, so the
            # opposite heading is refined too and competes as a separate candidate.
            for h in (heading, heading + np.pi):
                local = SearchRegion((x, y), 3 * gsd + 4.0, float(wrap_angle(h)), step)
                fix = self._local(gray, frame, local, scale)
                if fix is not None:
                    fixes.append(fix)
        fixes.sort(key=lambda f: f.score, reverse=True)
        return _with_distinctiveness(fixes, cfg.peak_separation_m)

    def _local(self, gray, frame, region, scale) -> Fix | None:
        cfg = self.cfg
        step = np.radians(cfg.fine_heading_step_deg)
        headings = _heading_grid(region.heading, region.heading_span, step)
        scales = tuple(scale * s for s in cfg.scales)
        res = self._search(
            gray, frame, cfg.fine_level, region.center, region.radius, headings, scales, keep=True
        )
        if res is None:
            return None
        r, c = np.unravel_index(int(np.argmax(res.best)), res.best.shape)
        s1 = float(res.best[r, c])
        if s1 <= 0:
            return None
        hi, si = np.unravel_index(int(res.best_idx[r, c]), (len(headings), len(scales)))
        peak = res.stack[hi, si]
        rr, cc = subpixel(peak, r, c)

        at = res.stack[:, :, r, c]
        dh = _neighbour_offset(at[:, si], hi)
        ds = _neighbour_offset(at[hi, :], si)
        heading = float(wrap_angle(headings[hi] + dh * step))
        scale_est = _interp(scales, si, ds)
        tel = frame.telemetry
        fp = footprint(self.camera, tel.altitude * scale_est, tel.pitch, tel.roll)
        x, y = self._to_world(res, rr, cc, heading, fp)

        _, gsd = self._features(cfg.fine_level)
        peaks = find_peaks(res.best, 2, max(int(cfg.sidelobe_m / gsd), 2))
        s2 = peaks[1][2] if len(peaks) > 1 else 0.0
        return Fix(x, y, heading, s1, _distinct(s1, s2), scale_est)

    def _search(self, gray, frame, level, center, radius, headings, scales, keep) -> _Search | None:
        tel = frame.telemetry
        feat, gsd = self._features(level)
        fps = {s: footprint(self.camera, tel.altitude * s, tel.pitch, tel.roll) for s in scales}
        radius_m = min(min(fp.radius for fp in fps.values()), self.cfg.max_radius_m)
        R = int(radius_m / gsd)
        if R < 8:
            return None
        mask_t = disk_mask(R)

        H, W = feat.shape
        if center is None or not np.isfinite(radius):
            u0, v0, u1, v1 = -R, -R, W + R, H + R
        else:
            # ``center`` is the aircraft; the template is centred on the optical-axis
            # ground point, which sits ahead of it when the camera is tilted.
            offsets = [rot2(h) @ fps[s].center for h in headings for s in scales]
            px = self.map.world_to_px(np.asarray(center) + np.array(offsets), level)
            lo = np.floor(px.min(0) - radius / gsd - R).astype(int)
            hi = np.ceil(px.max(0) + radius / gsd + R).astype(int) + 1
            (u0, v0), (u1, v1) = lo, hi
        roi, valid = _crop(feat, u0, v0, u1, v1)
        if roi.shape[0] < mask_t.shape[0] or roi.shape[1] < mask_t.shape[1] or not valid.any():
            return None
        corr = Correlator(roi, valid, mask_t, self.cfg.min_overlap)

        # One score per template offset, maximized over all hypotheses.
        best = np.full((roi.shape[0] - 2 * R, roi.shape[1] - 2 * R), -1.0, np.float32)
        best_idx = np.zeros(best.shape, np.int32)
        stack = np.empty((len(headings), len(scales), *best.shape), np.float32) if keep else None
        for hi, heading in enumerate(headings):
            for si, s in enumerate(scales):
                patch = rectify(
                    gray,
                    self.camera,
                    tel.altitude * s,
                    tel.pitch,
                    tel.roll,
                    heading,
                    gsd,
                    R,
                    fps[s],
                )
                score = corr.score(orientation_field(patch, mask_t, self.cfg.features))
                better = score > best
                best[better] = score[better]
                best_idx[better] = hi * len(scales) + si
                if stack is not None:
                    stack[hi, si] = score
        return _Search(best, stack, best_idx, headings, scales, fps, (u0, v0), R, level)

    def _to_world(self, res: _Search, r, c, heading: float, fp: Footprint):
        """Aircraft position for a template top-left offset ``(r, c)`` in the ROI.

        The patch centre is always the optical-axis ground point, whatever
        scale it was rendered at; ``fp`` should use the best altitude estimate.
        """
        u = res.origin[0] + c + res.radius_px
        v = res.origin[1] + r + res.radius_px
        center = self.map.px_to_world(np.array([u, v]), res.level)
        x, y = nadir_from_patch_center(center, heading, fp)
        return float(x), float(y)


# -- helpers ---------------------------------------------------------------


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _heading_grid(center: float | None, span: float, step: float) -> np.ndarray:
    if center is None or span >= np.pi:
        return np.arange(0.0, 2 * np.pi, step)
    n = max(int(np.ceil(span / step)), 1)
    return wrap_angle(center + step * np.arange(-n, n + 1))


def _crop(img: np.ndarray, u0: int, v0: int, u1: int, v1: int):
    """Crop ``img[v0:v1, u0:u1]``, zero-padding outside the image. Also returns validity."""
    H, W = img.shape
    out = np.zeros((v1 - v0, u1 - u0), img.dtype)
    valid = np.zeros(out.shape, bool)
    su0, sv0, su1, sv1 = max(u0, 0), max(v0, 0), min(u1, W), min(v1, H)
    if su0 < su1 and sv0 < sv1:
        out[sv0 - v0 : sv1 - v0, su0 - u0 : su1 - u0] = img[sv0:sv1, su0:su1]
        valid[sv0 - v0 : sv1 - v0, su0 - u0 : su1 - u0] = True
    return out, valid


def _neighbour_offset(scores: np.ndarray, i: int) -> float:
    if i == 0 or i == len(scores) - 1:
        return 0.0
    return parabola_offset(scores[i - 1], scores[i], scores[i + 1])


def _interp(values: tuple[float, ...], i: int, offset: float) -> float:
    if offset == 0.0 or len(values) == 1:
        return values[i]
    j = i + (1 if offset > 0 else -1)
    return values[i] + abs(offset) * (values[j] - values[i])


def _distinct(s1: float, s2: float) -> float:
    return float(np.clip(1.0 - max(s2, 0.0) / s1, 0.0, 1.0)) if s1 > 0 else 0.0


def _with_distinctiveness(fixes: list[Fix], separation: float) -> list[Fix]:
    """Set each fix's distinctiveness against its best rival: elsewhere, or turned around."""
    out = []
    for f in fixes:
        rivals = [
            g.score
            for g in fixes
            if g is not f
            and (
                np.hypot(g.x - f.x, g.y - f.y) >= separation
                or abs(wrap_angle(g.heading - f.heading)) >= np.pi / 2
            )
        ]
        d = _distinct(f.score, max(rivals, default=0.0))
        out.append(Fix(f.x, f.y, f.heading, f.score, d, f.scale))
    return out
