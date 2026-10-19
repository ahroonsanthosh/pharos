"""Render a :class:`World` under a given season, sun and atmosphere."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pharos.sim.procedural import fbm
from pharos.sim.world import N_CROPS, Cover, World

SEASONS = ("spring", "summer", "autumn", "winter", "snow")


@dataclass(frozen=True)
class Look:
    season: str = "summer"
    sun_azimuth_deg: float = 160.0  # compass direction towards the sun
    sun_elevation_deg: float = 50.0
    exposure: float = 1.0
    contrast: float = 1.0
    gamma: float = 1.0
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)  # RGB gains
    haze: float = 0.0  # 0..1 veil towards the sky colour
    clouds: float = 0.0  # fraction of the scene under cloud
    overcast: float = 0.0  # 0..1: diffuse light, so no shadows or sun-lit relief
    pasture: float = 0.0  # tone differences between grass fields (grazing, silage cuts)
    noise: float = 0.01  # per-pixel Gaussian noise (0..1 scale)
    seed: int = 0  # look-specific variation (soil moisture, crop tone)


# RGB, per season, indexed by Cover.
_COVER = {
    "spring": [
        (0.40, 0.60, 0.30),
        (0, 0, 0),
        (0.25, 0.42, 0.20),
        (0.15, 0.22, 0.25),
        (0.35, 0.35, 0.36),
        (0, 0, 0),
        (0.55, 0.47, 0.38),
        (0.60, 0.55, 0.45),
        (0.22, 0.40, 0.18),
        (0.40, 0.37, 0.27),
        (0.80, 0.76, 0.62),
    ],
    "summer": [
        (0.50, 0.58, 0.32),
        (0, 0, 0),
        (0.18, 0.32, 0.15),
        (0.13, 0.20, 0.22),
        (0.35, 0.35, 0.36),
        (0, 0, 0),
        (0.60, 0.52, 0.40),
        (0.62, 0.57, 0.46),
        (0.17, 0.30, 0.14),
        (0.42, 0.40, 0.27),
        (0.84, 0.80, 0.66),
    ],
    "autumn": [
        (0.48, 0.52, 0.30),
        (0, 0, 0),
        (0.50, 0.36, 0.17),
        (0.15, 0.21, 0.24),
        (0.33, 0.33, 0.34),
        (0, 0, 0),
        (0.46, 0.38, 0.30),
        (0.55, 0.50, 0.41),
        (0.45, 0.34, 0.18),
        (0.50, 0.36, 0.22),
        (0.78, 0.74, 0.60),
    ],
    "winter": [
        (0.50, 0.50, 0.38),
        (0, 0, 0),
        (0.33, 0.30, 0.27),
        (0.20, 0.25, 0.28),
        (0.34, 0.34, 0.35),
        (0, 0, 0),
        (0.42, 0.36, 0.30),
        (0.52, 0.48, 0.40),
        (0.32, 0.29, 0.26),
        (0.36, 0.31, 0.26),
        (0.72, 0.69, 0.58),
    ],
    "snow": [
        (0.92, 0.92, 0.95),
        (0, 0, 0),
        (0.42, 0.42, 0.43),
        (0.78, 0.80, 0.84),
        (0.50, 0.50, 0.53),
        (0, 0, 0),
        (0.90, 0.90, 0.93),
        (0.88, 0.88, 0.91),
        (0.40, 0.40, 0.41),
        (0.85, 0.85, 0.88),
        (0.80, 0.80, 0.83),
    ],
}

# RGB per crop type: wheat, maize, rapeseed, barley, grass ley, sugar beet.
_CROPS = {
    "spring": [
        (0.38, 0.55, 0.25),
        (0.55, 0.45, 0.35),
        (0.85, 0.80, 0.20),
        (0.45, 0.60, 0.30),
        (0.35, 0.55, 0.25),
        (0.50, 0.42, 0.32),
    ],
    "summer": [
        (0.78, 0.68, 0.42),
        (0.30, 0.50, 0.20),
        (0.50, 0.50, 0.30),
        (0.75, 0.70, 0.50),
        (0.40, 0.55, 0.28),
        (0.28, 0.48, 0.22),
    ],
    "autumn": [
        (0.68, 0.60, 0.45),
        (0.60, 0.55, 0.38),
        (0.42, 0.36, 0.28),
        (0.45, 0.38, 0.30),
        (0.42, 0.50, 0.30),
        (0.32, 0.45, 0.25),
    ],
    "winter": [
        (0.40, 0.45, 0.30),
        (0.50, 0.45, 0.35),
        (0.36, 0.30, 0.25),
        (0.38, 0.33, 0.27),
        (0.44, 0.48, 0.33),
        (0.40, 0.36, 0.30),
    ],
    "snow": [(0.90, 0.90, 0.93)] * N_CROPS,
}

# Contrast of crop rows / plough furrows per season.
_ROW_AMPLITUDE = {"spring": 0.10, "summer": 0.06, "autumn": 0.16, "winter": 0.12, "snow": 0.04}
_ROOFS = [(0.60, 0.25, 0.20), (0.42, 0.42, 0.44), (0.25, 0.25, 0.28), (0.70, 0.68, 0.65)]
_SKY = np.array([0.75, 0.80, 0.86], np.float32)


def render(world: World, look: Look) -> np.ndarray:
    """BGR uint8 image of the world at its native ground sample distance."""
    if look.season not in SEASONS:
        raise ValueError(f"unknown season {look.season!r}; choose from {SEASONS}")
    rng = np.random.default_rng(look.seed)
    h, w = world.shape
    px = 1.0 / world.gsd
    cover = world.cover
    snow = look.season == "snow"

    img = np.asarray(_COVER[look.season], np.float32)[cover]

    # Fields: per-parcel crop colour with tone jitter and crop-row texture.
    is_crop = cover == Cover.CROP
    tone = 1.0 + 0.07 * rng.standard_normal(len(world.crop)).astype(np.float32)
    crop_rgb = np.asarray(_CROPS[look.season], np.float32)[world.crop] * tone[:, None]
    img[is_crop] = crop_rgb[world.parcel[is_crop]]
    angle = world.rows[:, 0][world.parcel]
    period = world.rows[:, 1][world.parcel] * px
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    phase = (u * np.cos(angle) + v * np.sin(angle)) / period
    rows = 1.0 + _ROW_AMPLITUDE[look.season] * np.sin(2 * np.pi * phase)
    img *= np.where(is_crop, rows, 1.0)[..., None]
    del angle, period, phase, rows

    # Soil: persistent pattern plus look-specific moisture.
    moisture = fbm((h, w), rng, 120 * px, octaves=4)
    img *= (1.0 + (0.02 if snow else 0.07) * world.soil + 0.04 * moisture)[..., None]

    # Pasture: each grass field its own tone (grazing, silage cuts), some freshly mown.
    if look.pasture > 0:
        grass = cover == Cover.GRASS
        n = len(world.parcel_kind)
        tone = 1.0 + look.pasture * rng.standard_normal(n).astype(np.float32)
        mown = rng.random(n) < 0.5 * look.pasture / 0.1
        stripes = 1.0 + 0.05 * np.sign(np.sin(2 * np.pi * (u * 0.6 + v * 0.8) / (6 * px)))
        field = tone[world.parcel] * np.where(mown[world.parcel], stripes, 1.0)
        img *= np.where(grass, field, 1.0)[..., None]
        del field, stripes

    # Bog: heather and peat in patches.
    bog = cover == Cover.BOG
    if bog.any():
        img *= np.where(bog, 0.88 + 0.12 * np.tanh(world.canopy), 1.0)[..., None]

    # Trees: crown texture with sun-side highlights; under overcast skies, little of either.
    sun_az, sun_el = np.radians(look.sun_azimuth_deg), np.radians(look.sun_elevation_deg)
    sun_uv = np.array([np.sin(sun_az), -np.cos(sun_az)], np.float32)  # towards the sun, pixels
    direct = 1.0 - look.overcast
    trees = np.isin(cover, [Cover.FOREST, Cover.HEDGE])
    gu = cv2.Sobel(world.canopy, cv2.CV_32F, 1, 0, ksize=3)
    gv = cv2.Sobel(world.canopy, cv2.CV_32F, 0, 1, ksize=3)
    relief = (gu * sun_uv[0] + gv * sun_uv[1]) * (0.6 * np.cos(sun_el) * direct)
    crowns = 0.8 + 0.25 * np.tanh(world.canopy) + np.clip(relief, -0.35, 0.35)
    img *= np.where(trees, crowns, 1.0)[..., None]
    del gu, gv, relief, crowns

    # Buildings: two-pitch roofs lit according to the sun.
    roofs = np.zeros((h, w), np.uint8)
    for b in world.buildings:
        rgb = np.array((0.93, 0.93, 0.95) if snow else _ROOFS[b.roof], np.float32)
        c = b.corners
        mid_a, mid_b = (c[0] + c[3]) / 2, (c[1] + c[2]) / 2
        halves = (np.array([c[0], c[1], mid_b, mid_a]), np.array([mid_a, mid_b, c[2], c[3]]))
        ridge = c[1] - c[0]
        normal = np.array([-ridge[1], ridge[0]]) / (np.linalg.norm(ridge) + 1e-9)
        facing = float(normal @ sun_uv) * np.cos(sun_el) * direct
        for half, sign in zip(halves, (-1.0, 1.0), strict=True):
            shade = 1.0 + 0.35 * sign * facing
            poly = np.round(half).astype(np.int32)
            cv2.fillPoly(img, [poly], (rgb * shade).tolist())
            cv2.fillPoly(roofs, [poly], 1)

    # Cast shadows from buildings and trees.
    shadow = np.zeros((h, w), np.uint8)
    tan_el = max(np.tan(sun_el), 0.15)
    away = -sun_uv
    for b in world.buildings:
        d = away * min(b.height / tan_el, 60.0) * px
        hull = cv2.convexHull(np.round(np.vstack([b.corners, b.corners + d])).astype(np.int32))
        cv2.fillPoly(shadow, [hull], 1)
    for cls, height in ((Cover.FOREST, 15.0), (Cover.HEDGE, 3.0)):
        shadow |= _sweep(cover == cls, away * min(height / tan_el, 60.0) * px)
    shadow[(roofs > 0) | trees] = 0
    shade = cv2.GaussianBlur(shadow.astype(np.float32), (0, 0), 0.7 * px)
    img *= (1.0 - 0.55 * direct * shade)[..., None]
    del shadow, shade

    # Illumination, atmosphere, sensor response.
    sunlit = (0.55 + 0.45 * np.sin(sun_el)) / (0.55 + 0.45 * np.sin(np.radians(60)))
    img *= look.exposure * (direct * sunlit + 1.0 - direct)  # overcast light: as at 60 deg
    img *= np.asarray(look.white_balance, np.float32)
    if look.clouds > 0:
        _add_clouds(img, rng, look.clouds, px, away)
    if look.haze > 0:
        veil = np.clip(look.haze * (1.0 + 0.3 * fbm((h, w), rng, 300 * px, octaves=3)), 0, 1)
        img = img * (1 - veil[..., None]) + _SKY * veil[..., None]
    img = np.clip((img - 0.5) * look.contrast + 0.5, 0, 1) ** look.gamma
    if look.noise > 0:
        img += rng.normal(0, look.noise, img.shape).astype(np.float32)
    rgb = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _sweep(mask: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Union of ``mask`` translated along the segment from 0 to ``d`` (pixels)."""
    h, w = mask.shape
    m = mask.astype(np.uint8)
    out = np.zeros_like(m)
    for f in np.linspace(0, 1, max(int(np.linalg.norm(d) / 2), 1) + 1)[1:]:
        shift = np.float32([[1, 0, f * d[0]], [0, 1, f * d[1]]])
        out |= cv2.warpAffine(m, shift, (w, h), flags=cv2.INTER_NEAREST)
    return out


def _add_clouds(img: np.ndarray, rng, amount: float, px: float, away: np.ndarray) -> None:
    h, w = img.shape[:2]
    field = fbm((h, w), rng, 400 * px, octaves=5)
    thr = np.quantile(field, 1.0 - amount)
    cloud = np.clip((field - thr) / 0.6, 0, 1).astype(np.float32)
    d = away * 80 * px
    cast = cv2.warpAffine(cloud, np.float32([[1, 0, d[0]], [0, 1, d[1]]]), (w, h))
    img *= (1.0 - 0.45 * cast)[..., None]
    img *= (1.0 - cloud)[..., None]
    img += 0.95 * cloud[..., None]
