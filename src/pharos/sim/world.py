"""Procedural landscape geometry.

A :class:`World` describes *what is on the ground* (land cover, field
parcels, roads, buildings, trees) independently of how it looks. The same
world is rendered with different :class:`~pharos.sim.appearance.Look` s to
produce the reference "satellite" map and the drone's view, and
:func:`apply_changes` models what changed on the ground between the two
acquisitions (crop rotation, construction, felling).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import cv2
import numpy as np

from pharos.sim.procedural import catmull_rom, fbm


class Cover(IntEnum):
    GRASS = 0
    CROP = 1
    FOREST = 2
    WATER = 3
    ROAD = 4
    BUILDING = 5
    BARE = 6
    TRACK = 7
    HEDGE = 8


class Parcel(IntEnum):
    CROP = 0
    MEADOW = 1
    FOREST = 2
    BARE = 3


N_CROPS = 6
N_ROOFS = 4


@dataclass(frozen=True)
class WorldSpec:
    size_m: float = 1600.0
    gsd: float = 0.5
    parcel_spacing_m: float = 110.0
    forest_fraction: float = 0.12
    meadow_fraction: float = 0.15
    bare_fraction: float = 0.05
    hedge_fraction: float = 0.3
    track_fraction: float = 0.25
    major_roads: int = 3
    minor_road_fraction: float = 0.15
    river: bool = True
    ponds: int = 3
    villages: int = 4
    buildings_per_village: int = 35


@dataclass(frozen=True)
class ChangeSpec:
    """What changed on the ground between the map acquisition and the flight."""

    replant: float = 0.3  # fraction of crop parcels with a new crop and row direction
    buildings_removed: float = 0.0
    buildings_added: int = 0
    forest_cleared: float = 0.0  # fraction of forest parcels felled


@dataclass(frozen=True)
class Building:
    corners: np.ndarray  # (4, 2) pixel coordinates, ridge along corners[0] -> corners[1]
    height: float  # metres
    roof: int


@dataclass
class World:
    gsd: float
    cover: np.ndarray  # (H, W) uint8 Cover
    parcel: np.ndarray  # (H, W) int32 parcel id
    parcel_kind: np.ndarray  # (n,) Parcel
    crop: np.ndarray  # (n,) crop type per parcel
    rows: np.ndarray  # (n, 2) crop-row angle (rad) and period (m) per parcel
    buildings: list[Building]
    villages: list[tuple[float, float, float]]  # (u, v, street angle)
    canopy: np.ndarray  # (H, W) tree-crown texture, persistent
    soil: np.ndarray  # (H, W) low-frequency soil variation, persistent

    @property
    def shape(self) -> tuple[int, int]:
        return self.cover.shape


def generate_world(spec: WorldSpec, seed: int) -> World:
    rng = np.random.default_rng(seed)
    px = 1.0 / spec.gsd
    n = round(spec.size_m * px)
    shape = (n, n)

    parcel, facets = _parcels(shape, spec.parcel_spacing_m * px, rng)
    n_parcels = len(facets)
    kind = rng.choice(
        [Parcel.CROP, Parcel.MEADOW, Parcel.FOREST, Parcel.BARE],
        size=n_parcels,
        p=_normalize(
            [
                1 - spec.forest_fraction - spec.meadow_fraction - spec.bare_fraction,
                spec.meadow_fraction,
                spec.forest_fraction,
                spec.bare_fraction,
            ]
        ),
    ).astype(np.uint8)
    cover = np.array([Cover.CROP, Cover.GRASS, Cover.FOREST, Cover.BARE], np.uint8)[kind[parcel]]

    blobs = fbm(shape, rng, 250 * px, octaves=4) > 1.3
    cover[blobs & np.isin(cover, [Cover.CROP, Cover.GRASS])] = Cover.FOREST

    for facet in facets:
        u = rng.random()
        if u < spec.hedge_fraction:
            cv2.polylines(cover, [facet], True, int(Cover.HEDGE), max(int(3 * px), 1))
        elif u < spec.hedge_fraction + spec.track_fraction:
            cv2.polylines(cover, [facet], True, int(Cover.TRACK), max(int(2.5 * px), 1))

    if spec.river:
        _river(cover, rng, px)
    for _ in range(spec.ponds):
        c = tuple(int(v) for v in rng.uniform(0.1, 0.9, 2) * n)
        axes = tuple(int(v) for v in rng.uniform(10, 35, 2) * px)
        cv2.ellipse(cover, c, axes, rng.uniform(0, 180), 0, 360, int(Cover.WATER), -1)

    major = [_major_road(n, rng) for _ in range(spec.major_roads)]
    for line in major:
        cv2.polylines(cover, [line], False, int(Cover.ROAD), max(int(9 * px), 1))
    for facet in facets:
        if rng.random() < spec.minor_road_fraction and len(facet) > 2:
            k = rng.integers(0, len(facet))
            seg = np.roll(facet, -k, axis=0)[: rng.integers(2, min(5, len(facet)) + 1)]
            cv2.polylines(cover, [seg], False, int(Cover.ROAD), max(int(5 * px), 1))

    villages = [_village_site(major, n, rng) for _ in range(spec.villages)]
    buildings: list[Building] = []
    for vu, vv, angle in villages:
        # gardens and lawns around the houses
        disk = np.zeros(shape, np.uint8)
        cv2.circle(disk, (int(vu), int(vv)), int(110 * px), 1, -1)
        cover[(disk > 0) & np.isin(cover, [Cover.CROP, Cover.BARE])] = Cover.GRASS
        for _ in range(spec.buildings_per_village):
            b = _place_building(cover, (vu, vv, angle), px, rng)
            if b is not None:
                buildings.append(b)

    rows = np.column_stack(
        [rng.uniform(0, np.pi, n_parcels), rng.uniform(1.5, 4.0, n_parcels)]
    ).astype(np.float32)
    return World(
        gsd=spec.gsd,
        cover=cover,
        parcel=parcel,
        parcel_kind=kind,
        crop=rng.integers(0, N_CROPS, n_parcels),
        rows=rows,
        buildings=buildings,
        villages=villages,
        canopy=fbm(shape, rng, 8 * px, octaves=3),
        soil=fbm(shape, rng, 200 * px, octaves=5),
    )


def apply_changes(world: World, spec: ChangeSpec, seed: int) -> World:
    """A copy of ``world`` as it looks after the changes in ``spec``."""
    rng = np.random.default_rng(seed)
    px = 1.0 / world.gsd
    cover = world.cover.copy()
    crop, rows = world.crop.copy(), world.rows.copy()

    crops = np.flatnonzero(world.parcel_kind == Parcel.CROP)
    replant = crops[rng.random(len(crops)) < spec.replant]
    crop[replant] = rng.integers(0, N_CROPS, len(replant))
    rows[replant, 0] = rng.uniform(0, np.pi, len(replant))

    forests = np.flatnonzero(world.parcel_kind == Parcel.FOREST)
    felled = forests[rng.random(len(forests)) < spec.forest_cleared]
    if len(felled):
        cover[np.isin(world.parcel, felled) & (cover == Cover.FOREST)] = Cover.BARE

    keep = rng.random(len(world.buildings)) >= spec.buildings_removed
    buildings = [b for b, k in zip(world.buildings, keep, strict=True) if k]
    for b, k in zip(world.buildings, keep, strict=True):
        if not k:
            cv2.fillPoly(cover, [np.round(b.corners).astype(np.int32)], int(Cover.BARE))
    for _ in range(spec.buildings_added):
        if not world.villages:
            break
        site = world.villages[rng.integers(len(world.villages))]
        b = _place_building(cover, site, px, rng, spread=1.3)
        if b is not None:
            buildings.append(b)
    return replace(world, cover=cover, crop=crop, rows=rows, buildings=buildings)


# -- generators ------------------------------------------------------------


def _normalize(p: list[float]) -> np.ndarray:
    a = np.clip(np.asarray(p, float), 0, None)
    return a / a.sum()


def _parcels(shape, spacing_px: float, rng) -> tuple[np.ndarray, list[np.ndarray]]:
    """Voronoi field parcels from a jittered grid of seeds."""
    h, w = shape
    g = np.arange(-spacing_px, max(h, w) + spacing_px, spacing_px)
    gx, gy = np.meshgrid(g, g)
    pts = (
        np.column_stack([gx.ravel(), gy.ravel()])
        + rng.uniform(-0.45, 0.45, (gx.size, 2)) * spacing_px
    )
    subdiv = cv2.Subdiv2D(
        (
            int(-3 * spacing_px),
            int(-3 * spacing_px),
            w + int(6 * spacing_px),
            h + int(6 * spacing_px),
        )
    )
    for p in pts:
        subdiv.insert((float(p[0]), float(p[1])))
    facets, _ = subdiv.getVoronoiFacetList([])
    parcel = np.zeros(shape, np.int32)
    polys = []
    for i, f in enumerate(facets):
        poly = np.round(f).astype(np.int32)
        cv2.fillPoly(parcel, [poly], i)
        polys.append(poly)
    return parcel, polys


def _major_road(n: int, rng) -> np.ndarray:
    """A gently curving road crossing the map from one edge to the opposite one."""
    a = rng.uniform(0, np.pi)
    d = np.array([np.cos(a), np.sin(a)])
    normal = np.array([-d[1], d[0]])
    c = rng.uniform(0.25, 0.75, 2) * n
    s = np.linspace(-0.8 * n, 0.8 * n, 7)
    ctrl = c + s[:, None] * d + (rng.normal(0, 0.06 * n, len(s)))[:, None] * normal
    return np.round(catmull_rom(ctrl)).astype(np.int32)


def _river(cover: np.ndarray, rng, px: float) -> None:
    n = cover.shape[0]
    pos = np.array([0.0, rng.uniform(0.2, 0.8) * n])
    base = rng.uniform(-0.4, 0.4)
    heading, width = base, rng.uniform(12, 20)
    path = [pos.copy()]
    widths = [width]
    while 0 <= pos[0] < n and -0.1 * n < pos[1] < 1.1 * n and len(path) < 5000:
        heading += rng.normal(0, 0.12) - 0.08 * (heading - base)
        pos += 10 * px * np.array([np.cos(heading), np.sin(heading)])
        width = float(np.clip(width + rng.normal(0, 1.0), 10, 28))
        path.append(pos.copy())
        widths.append(width)
    pts = np.round(np.array(path)).astype(np.int32)
    # riparian strip of trees, drawn first so the water sits on top
    for i in range(len(pts) - 1):
        th = int((widths[i] + 16) * px)
        if rng.random() < 0.6:
            cv2.line(cover, tuple(pts[i]), tuple(pts[i + 1]), int(Cover.FOREST), th)
    for i in range(len(pts) - 1):
        cv2.line(cover, tuple(pts[i]), tuple(pts[i + 1]), int(Cover.WATER), int(widths[i] * px))


def _village_site(roads: list[np.ndarray], n: int, rng) -> tuple[float, float, float]:
    """A village centre on a major road (if any), with the local street direction."""
    for _ in range(50):
        if roads:
            line = roads[rng.integers(len(roads))]
            i = int(rng.integers(1, len(line) - 1))
            u, v = line[i]
            du, dv = line[i + 1] - line[i - 1]
            angle = float(np.arctan2(dv, du))
        else:
            u, v = rng.uniform(0, n, 2)
            angle = float(rng.uniform(0, np.pi))
        if 0.1 * n < u < 0.9 * n and 0.1 * n < v < 0.9 * n:
            return float(u), float(v), angle
    return n / 2, n / 2, 0.0


def _place_building(cover, site, px: float, rng, spread: float = 1.0) -> Building | None:
    vu, vv, angle = site
    for _ in range(20):
        r = abs(rng.normal(0, 55 * spread)) * px
        t = rng.uniform(0, 2 * np.pi)
        cu, cv = vu + r * np.cos(t), vv + r * np.sin(t)
        length, width = rng.uniform(9, 22) * px, rng.uniform(7, 12) * px
        a = angle + rng.choice([0.0, np.pi / 2]) + rng.normal(0, np.radians(4))
        d = np.array([np.cos(a), np.sin(a)])
        nrm = np.array([-d[1], d[0]])
        c = np.array([cu, cv])
        corners = np.array(
            [
                c - d * length / 2 - nrm * width / 2,
                c + d * length / 2 - nrm * width / 2,
                c + d * length / 2 + nrm * width / 2,
                c - d * length / 2 + nrm * width / 2,
            ]
        )
        if _footprint_clear(cover, corners, margin=int(3 * px)):
            cv2.fillPoly(cover, [np.round(corners).astype(np.int32)], int(Cover.BUILDING))
            return Building(corners, float(rng.uniform(4, 12)), int(rng.integers(N_ROOFS)))
    return None


def _footprint_clear(cover: np.ndarray, corners: np.ndarray, margin: int) -> bool:
    h, w = cover.shape
    x0, y0 = np.floor(corners.min(0)).astype(int) - margin
    x1, y1 = np.ceil(corners.max(0)).astype(int) + margin + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return False
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round(corners - [x0, y0]).astype(np.int32)], 1)
    if margin:
        m = cv2.dilate(m, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    blocked = np.isin(cover[y0:y1, x0:x1], [Cover.ROAD, Cover.WATER, Cover.BUILDING])
    return not np.any(blocked & (m > 0))
