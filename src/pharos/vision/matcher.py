"""Frame-to-map matchers: turn one frame into absolute position hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cache
from typing import Protocol

import cv2
import numpy as np

from pharos.geomap import GeoMap
from pharos.geometry import Camera, rot2, wrap_angle
from pharos.types import Fix, Frame, HeightField, Telemetry
from pharos.vision.correlate import (
    Correlator,
    find_peaks,
    parabola_offset,
    peak_covariance,
    peak_excess,
    subpixel,
)
from pharos.vision.features import FeatureConfig, orientation_field
from pharos.vision.rectify import (
    Footprint,
    disk_mask,
    footprint,
    frame_mask,
    frame_pixels,
    nadir_from_patch_center,
    patch_in_frame,
    rectify,
)


@dataclass(frozen=True)
class SearchRegion:
    """Where to look. ``center=None`` searches the whole map; ``heading=None`` all headings."""

    center: tuple[float, float] | None = None
    radius: float = float("inf")
    heading: float | None = None
    heading_span: float = np.pi  # half-width, radians


class Matcher(Protocol):
    def match(
        self,
        frame: Frame,
        region: SearchRegion,
        scale: float = 1.0,
        height: HeightField | None = None,
    ) -> list[Fix]:
        """Best-first absolute fixes for ``frame`` inside ``region``.

        ``scale`` multiplies the height above ground (the caller's running
        estimate of the barometric bias). Over uneven ground ``height`` gives
        that height wherever the aircraft might be, instead of the telemetry.
        """
        ...


@dataclass(frozen=True)
class MatcherConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    fine_level: int = 0  # map pyramid level for tracking and refinement
    coarse_level: int = 1  # map pyramid level for the global search
    # Low down a frame sees little ground, and on a coarse map too few pixels to match:
    # each search then takes the coarsest level with at least this many pixels across the
    # frame's half-width (its footprint radius), but never one coarser than fine_level
    # (coarse_level for wide searches) nor finer than the map. Simulated pasture at 60 m
    # was located for 68% of the flight against a 1 m map and 80% against a 0.5 m one.
    min_radius_px: float = 45.0
    max_radius_m: float = 160.0  # cap on the matched disk radius
    global_radius_m: float = 400.0  # regions wider than this use the coarse global search
    coarse_heading_step_deg: float = 6.0
    fine_heading_step_deg: float = 2.0
    scales: tuple[float, ...] = (0.94, 1.0, 1.06)  # altitude-correction hypotheses
    # Over uneven ground the global search renders the frame for heights this ratio apart,
    # each scored only where the terrain puts the aircraft at that height, max_height_bins
    # at a time: the expected one and those nearest it first, the others in turn (when a
    # terrain-following flight's barometer drifts from its height above ground, the right
    # one can be far off). The coarse search finds the place with the height 10% off.
    height_step: float = 0.2
    max_height_bins: int = 5
    # Local searches over uneven ground render the frame through the terrain, not onto the
    # plane under its centre (relief averaged over the position's uncertainty).
    terrain_relief: bool = True
    candidates: int = 5  # coarse candidates refined in a global search
    peak_separation_m: float = 40.0  # peaks closer than this are the same place
    sidelobe_m: float = 8.0  # tracking: rival peaks must be at least this far from the best
    min_overlap: float = 0.75
    open_ridge_m: float = 150.0  # along-ridge uncertainty of a ridge leaving the search window
    # "frame": all the ground the image covers; "disk": the largest disk inside it, the same
    # for every heading. On real 16:9 video at 80 m the frame covers 2.3 times the ground,
    # and the true location stood out in 62% of frames instead of 40%.
    template: str = "frame"
    # The coarse (global) level reduces features computed at the fine level, instead of
    # computing them on a reduced image: hedges, drains and walls a few metres wide keep
    # their direction at 2 m per pixel instead of blurring away.
    fine_features: bool = True


@dataclass
class _Search:
    """Result of scoring every (heading, height) hypothesis over one ROI."""

    best: np.ndarray  # max score over hypotheses, per template offset
    stack: np.ndarray | None  # (n_headings, n_altitudes, H, W) scores if kept
    best_idx: np.ndarray  # flat hypothesis index of ``best``
    headings: np.ndarray
    altitudes: tuple[float, ...]  # heights above ground the frame was rendered for
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
        self._features = cache(self._compute_features)  # (level, coarse) -> field, gsd
        self._sweeps = 0  # wide searches over uneven ground so far (they take turns at heights)

    def _compute_features(self, level: int, coarse: bool) -> tuple[np.ndarray, float]:
        """Map features at a pyramid ``level``; for a wide search (``coarse``) the finer
        level's reduced (fine_features)."""
        if coarse and level > 0 and self.cfg.fine_features:
            field, gsd = self._features(level - 1, False)
            return _reduce(field), 2 * gsd
        img, gsd = self.map.level(level)
        return orientation_field(img, None, self.cfg.features), gsd

    def _levels(self, frame: Frame, altitude: float) -> tuple[int, int]:
        """Map levels to match ``frame`` at from ``altitude``: fine and wide-search."""
        cfg, tel = self.cfg, frame.telemetry
        fine, coarse = cfg.fine_level, cfg.coarse_level
        if cfg.min_radius_px <= 0 or not altitude > 0:
            return fine, coarse
        try:
            radius = footprint(self.camera, altitude, tel.pitch, tel.roll).radius
        except ValueError:  # tilted above the horizon
            return fine, coarse
        want = radius / cfg.min_radius_px  # metres per pixel that still give enough pixels
        finer = int(np.floor(np.log2(max(want, 1e-6) / self.map.gsd) + 1e-9))
        step = max(fine - max(finer, 0), 0)  # levels finer than configured, if any
        return fine - step, coarse - step

    def match(
        self,
        frame: Frame,
        region: SearchRegion,
        scale: float = 1.0,
        height: HeightField | None = None,
    ) -> list[Fix]:
        gray = _gray(frame.image)
        if region.center is None or region.radius > self.cfg.global_radius_m:
            return self._global(gray, frame, region, scale, height)
        fix = self._local(gray, frame, region, scale, height)
        return [] if fix is None else [fix]

    # -- search modes -----------------------------------------------------

    def _global(self, gray, frame, region, scale, height) -> list[Fix]:
        cfg = self.cfg
        step = np.radians(cfg.coarse_heading_step_deg)
        expected = frame.telemetry.altitude * scale
        altitudes, select = (expected,), None
        if height is not None:
            altitudes, select = self._height_bins(height, scale, region, expected)
        _, coarse = self._levels(frame, expected)
        res = self._search(
            gray,
            frame,
            coarse,
            region.center,
            region.radius,
            _heading_grid(region.heading, region.heading_span, step),
            altitudes,
            keep=False,
            select=select,
            coarse=True,
        )
        if res is None:
            return []
        _, gsd = self._features(coarse, True)
        sep = max(int(cfg.peak_separation_m / gsd), 1)
        fixes = []
        for r, c, _ in find_peaks(res.best, cfg.candidates, sep):
            hi, ai = divmod(int(res.best_idx[r, c]), len(altitudes))
            heading = float(res.headings[hi])
            x, y = self._to_world(res, r, c, heading, res.footprints[altitudes[ai]])
            # Line-dominated scenes can match almost as well turned around, so the
            # opposite heading is refined too and competes as a separate candidate, unless
            # the region rules it out (a compass does: on a real flight a turned-around
            # candidate won after a U-turn and the track flew backwards for minutes).
            turned = heading + np.pi
            for h in (heading, turned) if _allows(region, turned) else (heading,):
                local = SearchRegion((x, y), 3 * gsd + 4.0, float(wrap_angle(h)), step)
                fix = self._local(gray, frame, local, scale, height)
                if fix is not None:
                    fixes.append(fix)
        fixes.sort(key=lambda f: f.score, reverse=True)
        return _with_distinctiveness(fixes, cfg.peak_separation_m)

    def _local(self, gray, frame, region, scale, height) -> Fix | None:
        cfg = self.cfg
        step = np.radians(cfg.fine_heading_step_deg)
        headings = _heading_grid(region.heading, region.heading_span, step)
        tel = frame.telemetry
        base = tel.altitude
        if height is not None:
            spread = region.radius / 3  # the search covers three sigmas
            base = view_height(height, self.camera, tel, region.center, region.heading, spread)
            if not base > 0:  # underground here
                return None
        altitudes = tuple(base * scale * s for s in cfg.scales)
        relief = None
        if height is not None and cfg.terrain_relief and region.heading is not None:
            spread = region.radius / 3

            def relief(fp: Footprint, gsd: float, r: int) -> np.ndarray | None:
                return _relief(height, region.center, region.heading, base, spread, fp, gsd, r)

        fine, _ = self._levels(frame, base * scale)
        res = self._search(
            gray,
            frame,
            fine,
            region.center,
            region.radius,
            headings,
            altitudes,
            True,
            relief=relief,
        )
        if res is None:
            return None
        r, c = np.unravel_index(int(np.argmax(res.best)), res.best.shape)
        s1 = float(res.best[r, c])
        if s1 <= 0:
            return None
        hi, si = np.unravel_index(int(res.best_idx[r, c]), (len(headings), len(altitudes)))
        peak = res.stack[hi, si]
        rr, cc = subpixel(peak, r, c)

        at = res.stack[:, :, r, c]
        dh = _neighbour_offset(at[:, si], hi)
        ds = _neighbour_offset(at[hi, :], si)
        heading = float(wrap_angle(headings[hi] + dh * step))
        altitude = _interp(altitudes, si, ds)
        fp = footprint(self.camera, altitude, tel.pitch, tel.roll)
        x, y = self._to_world(res, rr, cc, heading, fp)

        _, gsd = self._features(fine, False)
        peaks = find_peaks(res.best, 2, max(int(cfg.sidelobe_m / gsd), 2))
        s2 = peaks[1][2] if len(peaks) > 1 else 0.0
        rho = 2 * cfg.features.smooth
        excess = peak_excess(res.best, r, c, rho)
        # rows run south: (row, col) pixels to (east, north) metres
        open_variance = (cfg.open_ridge_m / gsd) ** 2
        (crr, crc), (_, ccc) = peak_covariance(res.best, r, c, rho, open_variance) * gsd**2
        spread = (float(ccc), float(-crc), float(crr))
        return Fix(x, y, heading, s1, _distinct(s1, s2), altitude / base, excess, spread)

    def _height_bins(self, height: HeightField, scale: float, region: SearchRegion, expected):
        """Heights to render the frame for over ``region``, and which one applies where."""
        cfg = self.cfg
        x0, y0, x1, y1 = self.map.extent
        if region.center is not None and np.isfinite(region.radius):
            (cx, cy), r = region.center, region.radius
            x0, y0, x1, y1 = max(x0, cx - r), max(y0, cy - r), min(x1, cx + r), min(y1, cy + r)
        grid = np.stack(np.meshgrid(np.linspace(x0, x1, 50), np.linspace(y0, y1, 50)), -1)
        step = np.log1p(cfg.height_step)

        def bin_of(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            h = scale * height(xy)
            ok = h > 0
            return np.round(np.log(np.where(ok, h, expected) / expected) / step).astype(int), ok

        k, ok = bin_of(grid)
        nearest = sorted(set(k[ok].tolist()), key=abs)
        if not nearest:
            return (), None
        turns = -(-len(nearest) // cfg.max_height_bins)
        turn = self._sweeps % turns
        self._sweeps += 1
        chosen = sorted(nearest[turn * cfg.max_height_bins : (turn + 1) * cfg.max_height_bins])
        altitudes = tuple(float(expected * np.exp(step * b)) for b in chosen)
        lut = np.full(chosen[-1] - chosen[0] + 1, -1)
        lut[np.array(chosen) - chosen[0]] = np.arange(len(chosen))

        def select(xy: np.ndarray) -> np.ndarray:
            k, ok = bin_of(xy)
            inside = ok & (k >= chosen[0]) & (k <= chosen[-1])
            return np.where(inside, lut[np.clip(k - chosen[0], 0, len(lut) - 1)], -1)

        return altitudes, select

    def _search(
        self,
        gray,
        frame,
        level,
        center,
        radius,
        headings,
        altitudes,
        keep,
        select=None,
        relief=None,
        coarse=False,
    ) -> _Search | None:
        """Score the frame rendered at each heading and height above ground over a region
        of map ``level`` (``coarse``: a wide search's).

        ``select``, given the world positions of the template centres (where the camera
        looks), says which height applies at each, -1 for none. ``relief(fp, gsd, r)`` gives
        the terrain under the heading-up patch (see :func:`_relief`), or None for a plane.
        """
        if not altitudes:
            return None
        tel = frame.telemetry
        feat, gsd = self._features(level, bool(coarse))
        fps = {a: footprint(self.camera, a, tel.pitch, tel.roll) for a in altitudes}
        smallest = fps[min(altitudes)]  # inside every height's coverage
        whole = self.cfg.template == "frame"
        steps = int(coarse and self.cfg.fine_features)
        made = self._templates(gray, frame, altitudes, fps, gsd, relief, steps)
        if made is None:
            return None
        R, masks, upright, own_mask = made  # own_mask: not a plane's frame_mask
        mask_t = disk_mask(R)

        H, W = feat.shape
        if center is None or not np.isfinite(radius):
            u0, v0, u1, v1 = -R, -R, W + R, H + R
        else:
            # ``center`` is the aircraft; the template is centred on the optical-axis
            # ground point, which sits ahead of it when the camera is tilted.
            offsets = [rot2(h) @ fp.center for h in headings for fp in fps.values()]
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
        allowed = None
        if select is not None:
            allowed = _selection(select, self.map, level, best.shape, (u0 + R, v0 + R))
            allowed = [allowed == ai for ai in range(len(altitudes))]
        best_idx = np.zeros(best.shape, np.int32)
        stack = np.empty((len(headings), len(altitudes), *best.shape), np.float32) if keep else None
        shared = len({id(m) for m in masks.values()}) == 1  # one mask for every height
        for hi, heading in enumerate(headings):
            corr_h = corr
            for ai, a in enumerate(altitudes):
                if ai == 0 or not shared:
                    if own_mask:
                        corr_h = corr.with_mask(_turned_mask(masks[a], heading))
                    elif whole:  # the covered ground turns with the heading, and so does its mask
                        fp = smallest if shared else fps[a]
                        corr_h = corr.with_mask(frame_mask(fp, heading, gsd, R))
                score = corr_h.score(_turned(upright[a], heading))
                if allowed is not None:
                    # Heights compete in how far they stand out from their own chance scores:
                    # a lower height's smaller template scores higher by chance alone.
                    score = _standardized(score, allowed[ai])
                better = score > best
                best[better] = score[better]
                best_idx[better] = hi * len(altitudes) + ai
                if stack is not None:
                    stack[hi, ai] = score
        return _Search(best, stack, best_idx, headings, altitudes, fps, (u0, v0), R, level)

    def _templates(self, gray, frame, altitudes, fps, gsd, relief, steps=0):
        """Heading-up templates of the frame: radius, mask and field per height, own mask.

        A heading only turns the patch about its centre, so features are computed once
        per height, heading-up, and turned for each heading hypothesis.
        """
        tel = frame.telemetry
        smallest = fps[min(altitudes)]
        whole = self.cfg.template == "frame"
        # Heights far apart (a wide search over uneven ground) each use all they see; heights
        # a few percent apart share the smallest one's coverage, which is inside the others'.
        apart = whole and max(altitudes) > 1.15 * min(altitudes)
        widest = fps[max(altitudes)] if apart else smallest
        reach = widest.extent if whole else widest.radius
        R = int(min(reach, self.cfg.max_radius_m) / gsd)
        if R < 8:
            return None
        mask_up = frame_mask(smallest, 0.0, gsd, R) if whole else disk_mask(R)
        masks = {a: frame_mask(fp, 0.0, gsd, R) if apart else mask_up for a, fp in fps.items()}
        dz = relief(fps[altitudes[len(altitudes) // 2]], gsd, R) if relief is not None else None
        if dz is not None:  # the frame covers what the terrain lets it see
            uv = frame_pixels(
                self.camera, min(altitudes), tel.pitch, tel.roll, 0.0, gsd, R, smallest, dz
            )
            mask_up = mask_up & patch_in_frame(uv, self.camera)
            masks = dict.fromkeys(fps, mask_up)
        upright = {
            a: self._field(gray, tel, a, gsd, R, fp, masks[a], dz, steps) for a, fp in fps.items()
        }
        return R, masks, upright, dz is not None

    def _field(self, gray, tel, altitude, gsd, r, fp, mask, dz=None, steps=0) -> np.ndarray:
        """Orientation field of the frame rectified at ``gsd`` (radius ``r``), computed
        ``steps`` pyramid levels finer and reduced, like the map's (fine_features)."""
        if steps == 0 or dz is not None:
            patch = rectify(gray, self.camera, altitude, tel.pitch, tel.roll, 0.0, gsd, r, fp, dz)
            return orientation_field(patch, mask, self.cfg.features)
        k = 2**steps
        fine = (2 * r) * k + 1
        patch = rectify(gray, self.camera, altitude, tel.pitch, tel.roll, 0.0, gsd / k, r * k, fp)
        big = cv2.resize(mask.astype(np.uint8), (fine, fine), interpolation=cv2.INTER_NEAREST)
        field = orientation_field(patch, big > 0, self.cfg.features)
        for _ in range(steps):
            field = _reduce(field)
        return np.where(mask, field, 0).astype(np.complex64)

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


# Unit offsets around a position, to average the terrain over its uncertainty.
_AROUND = np.array([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)])
_AROUND = _AROUND / np.maximum(np.hypot(*_AROUND.T), 1.0)[:, None]


def view_height(
    height: HeightField,
    camera: Camera,
    tel: Telemetry,
    xy,
    heading: float | None,
    spread: float = 0.0,
) -> float:
    """Height above the ground the camera looks at, for an aircraft at ``xy``.

    A tilted camera sees ground ahead of the aircraft, which may lie higher or
    lower. ``spread`` is the position's uncertainty (m): over a slope a position
    error is a height error, so the height is averaged over it. NaN if the
    aircraft would be underground everywhere.
    """
    points = np.asarray(xy, float) + spread * _AROUND if spread > 0 else np.asarray(xy, float)
    h = np.atleast_1d(height(points))
    if heading is not None and tel.tilt != 0.0 and (h > 0).any():
        ahead = footprint(camera, float(h[h > 0].mean()), tel.pitch, tel.roll).center
        h = np.atleast_1d(height(points + rot2(heading) @ ahead))
    h = h[h > 0]
    return float(h.mean()) if len(h) else float("nan")


def relief_at(height: HeightField, xy: np.ndarray, base: float, spread: float) -> np.ndarray:
    """The ground's height (m) at world positions ``(..., 2)`` above a plane ``base`` below
    the aircraft, averaged over ``spread`` (the position's uncertainty); 0 where unknown."""
    h = height(np.asarray(xy, float)[..., None, :] + spread * _AROUND)
    known = h > 0
    seen = known.sum(-1)
    mean = np.where(known, h, 0.0).sum(-1) / np.maximum(seen, 1)
    return np.where(seen > 0, base - mean, 0.0)


def _selection(select, geomap: GeoMap, level: int, shape, origin, stride: int = 4) -> np.ndarray:
    """``select`` at every template offset (centre ``origin`` + offset, level pixels).

    Evaluated every ``stride`` offsets: terrain changes slowly.
    """
    rows, cols = shape
    u = origin[0] + np.arange(0, cols, stride)
    v = origin[1] + np.arange(0, rows, stride)
    choice = select(geomap.px_to_world(np.stack(np.meshgrid(u, v), -1), level))
    return np.repeat(np.repeat(choice, stride, 0), stride, 1)[:rows, :cols]


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _turned(field: np.ndarray, heading: float) -> np.ndarray:
    """A heading-up orientation field as the north-up patch for ``heading`` would give it.

    Patch pixels turn by ``heading`` about the centre (clockwise on screen, rows run
    south), and the doubled-angle orientations with them.
    """
    if heading == 0.0:
        return field
    n = field.shape[0]
    M, flags = _turn(n, heading), cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    parts = [
        cv2.warpAffine(np.ascontiguousarray(p), M, (n, n), flags=flags)
        for p in (field.real, field.imag)
    ]
    return ((parts[0] + 1j * parts[1]) * np.exp(2j * heading)).astype(np.complex64)


def _standardized(score: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Scores where ``allowed`` as standard deviations above their mean there; -inf elsewhere."""
    out = np.full(score.shape, -np.inf, np.float32)
    ok = allowed & (score > -1.0)
    if ok.sum() > 1:
        values = score[ok]
        out[ok] = (values - values.mean()) / max(float(values.std()), 1e-6)
    return out


def _reduce(field: np.ndarray) -> np.ndarray:
    """A complex field at half the resolution, sampled like an image pyramid level."""
    parts = [cv2.pyrDown(np.ascontiguousarray(p)) for p in (field.real, field.imag)]
    return (parts[0] + 1j * parts[1]).astype(np.complex64)


def _turned_mask(mask: np.ndarray, heading: float) -> np.ndarray:
    """A heading-up patch mask turned like :func:`_turned` turns its field."""
    n = mask.shape[0]
    flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    return cv2.warpAffine(mask.astype(np.uint8), _turn(n, heading), (n, n), flags=flags) > 0


def _turn(n: int, heading: float) -> np.ndarray:
    """Affine map from a turned patch's pixels to the heading-up patch's."""
    r = (n - 1) / 2
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, s], [-s, c]])  # new pixel offset -> heading-up pixel offset
    return np.column_stack([rot, r - rot @ np.array([r, r])])


def _relief(
    height: HeightField,
    center,
    heading: float,
    base: float,
    spread: float,
    fp: Footprint,
    gsd: float,
    r: int,
    step: int = 8,
) -> np.ndarray | None:
    """The ground's height (m) above the plane under a heading-up patch's centre.

    For an aircraft at ``center`` with ``heading``, ``base`` above that plane. The
    terrain is averaged over ``spread`` (the position's uncertainty): relief finer
    than that is not known where it lies. Sampled every ``step`` patch pixels and
    interpolated; None where the ground is flat to 1% of the height.
    """
    n = 2 * r + 1
    m = max(int(np.ceil((n - 1) / step)) + 1, 2)
    k = np.linspace(0, n - 1, m)
    local = np.stack(np.meshgrid(fp.center[0] + (k - r) * gsd, fp.center[1] - (k - r) * gsd), -1)
    dz = relief_at(height, np.asarray(center, float) + local @ rot2(heading).T, base, spread)
    if np.abs(dz).max() < 0.01 * base:
        return None
    index = (np.arange(n) * (m - 1) / (n - 1)).astype(np.float32)
    mx, my = np.meshgrid(index, index)
    return cv2.remap(dz.astype(np.float32), mx, my, cv2.INTER_LINEAR)


def _allows(region: SearchRegion, heading: float) -> bool:
    """Whether ``heading`` is within ``region``'s heading span."""
    if region.heading is None or region.heading_span >= np.pi:
        return True
    return abs(float(wrap_angle(heading - region.heading))) <= region.heading_span


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
        out.append(replace(f, distinctiveness=d))
    return out
