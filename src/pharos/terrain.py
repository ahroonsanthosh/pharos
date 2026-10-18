"""Ground elevation under the map, so heights above takeoff become heights above ground.

Barometers measure height above the takeoff point. Over a valley the ground is
further away than that, over a hill closer, and the image scale follows the
ground. With a terrain model the height above ground anywhere is the
aircraft's elevation (the takeoff's plus the reported height) minus the
ground's there.

Copernicus GLO-30, a 30 m surface model of the whole world (free, no account,
AWS Open Data), is fetched for the map's area and cached after the first
download. Any other DEM GeoTIFF, such as a national lidar model, works the
same way. Both are surface models: over a forest they give the canopy top,
which is also what the camera sees. The takeoff elevation is read from the
same model, so its vertical datum cancels out.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from pharos.geomap import GeoMap
    from pharos.types import HeightField

log = logging.getLogger(__name__)

COPERNICUS_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{0}/{0}.tif"


@dataclass
class Terrain:
    """Elevation in metres on a north-up grid in a map's world coordinates.

    Node ``(row, col)`` sits at ``x = origin_x + col * spacing``,
    ``y = origin_y - row * spacing``.
    """

    elevation: np.ndarray  # float32, at least 2 x 2
    spacing: float
    origin_x: float
    origin_y: float

    def at(self, xy) -> np.ndarray:
        """Elevation at world positions ``(..., 2)``, bilinear; the edge value beyond the grid."""
        xy = np.asarray(xy, float)
        z = self.elevation
        h, w = z.shape
        known = np.isfinite(xy).all(-1)
        c = np.clip((np.where(known, xy[..., 0], 0.0) - self.origin_x) / self.spacing, 0, w - 1)
        r = np.clip((self.origin_y - np.where(known, xy[..., 1], 0.0)) / self.spacing, 0, h - 1)
        c0 = np.minimum(c.astype(int), w - 2)
        r0 = np.minimum(r.astype(int), h - 2)
        fc, fr = c - c0, r - r0
        top = z[r0, c0] * (1 - fc) + z[r0, c0 + 1] * fc
        bottom = z[r0 + 1, c0] * (1 - fc) + z[r0 + 1, c0 + 1] * fc
        return np.where(known, top * (1 - fr) + bottom * fr, np.nan)

    def height_above(self, elevation: float) -> HeightField:
        """Height above the ground anywhere of an aircraft at ``elevation`` (same datum)."""
        return lambda xy: elevation - self.at(xy)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            elevation=self.elevation,
            spacing=self.spacing,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
        )

    @classmethod
    def load(cls, path: str | Path) -> Terrain:
        with np.load(path) as f:
            return cls(
                f["elevation"], float(f["spacing"]), float(f["origin_x"]), float(f["origin_y"])
            )

    @classmethod
    def from_dem(
        cls,
        paths: Sequence[str | Path],
        geomap: GeoMap,
        include: Iterable[tuple[float, float]] = (),
        margin_m: float = 500.0,
        spacing: float = 10.0,
    ) -> Terrain:
        """DEM GeoTIFFs in any CRS, resampled onto ``geomap``'s world coordinates.

        The grid covers the map and the world positions ``include`` (a takeoff
        point, say) with ``margin_m`` to spare. Where several files overlap the
        first wins; points no file covers take the nearest covered value.
        """
        terrain, lat, lon = _grid(geomap, include, margin_m, spacing)
        z = terrain.elevation
        for path in paths:
            todo = np.isnan(z)
            if todo.any():
                z[todo] = _sample(path, lat[todo], lon[todo])
        if np.isnan(z).all():
            raise ValueError("the DEM does not cover the map")
        terrain.elevation = _fill_nearest(z)
        return terrain

    @classmethod
    def copernicus(
        cls,
        geomap: GeoMap,
        include: Iterable[tuple[float, float]] = (),
        margin_m: float = 500.0,
        spacing: float = 10.0,
        cache: Path | None = None,
    ) -> Terrain:
        """Copernicus GLO-30 for the map's area (see :meth:`from_dem`), fetched once and cached.

        Tiles are 1 by 1 degree, about 40 MB each, kept under ``cache``
        (default ``$PHAROS_CACHE`` or ``~/.cache/pharos``). There are no tiles
        over open sea, where the elevation is 0.
        """
        terrain, lat, lon = _grid(geomap, include, margin_m, spacing)
        z = terrain.elevation
        folder = (cache or cache_dir()) / "copernicus"
        south, west = np.floor(lat).astype(int), np.floor(lon).astype(int)
        for s, w in np.unique(np.column_stack([south.ravel(), west.ravel()]), axis=0):
            inside = (south == s) & (west == w)
            path = copernicus_tile(int(s), int(w), folder)
            z[inside] = 0.0 if path is None else _sample(path, lat[inside], lon[inside])
        terrain.elevation = _fill_nearest(z)
        return terrain


def cache_dir() -> Path:
    if root := os.environ.get("PHAROS_CACHE"):
        return Path(root)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "pharos"


def copernicus_tile(lat: int, lon: int, folder: Path) -> Path | None:
    """The cached GLO-30 tile whose south-west corner is (``lat``, ``lon``); None over open sea."""
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    path, sea = folder / f"{name}.tif", folder / f"{name}.sea"
    if path.exists():
        return path
    if sea.exists():
        return None
    folder.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    log.info("downloading %s", name)
    try:
        with (
            urllib.request.urlopen(COPERNICUS_URL.format(name), timeout=60) as r,
            open(part, "wb") as f,
        ):
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as exc:
        if exc.code not in (403, 404):
            raise
        sea.touch()  # the bucket has no tile here: open sea
        return None
    part.replace(path)
    return path


def _grid(geomap: GeoMap, include, margin_m: float, spacing: float):
    """An empty terrain grid around the map and ``include``, and its nodes' lat and lon."""
    x0, y0, x1, y1 = geomap.extent
    points = np.array([(x0, y0), (x1, y1), *include], float)
    lo = points.min(0) - margin_m
    hi = points.max(0) + margin_m
    cols = int(np.ceil((hi[0] - lo[0]) / spacing)) + 1
    rows = int(np.ceil((hi[1] - lo[1]) / spacing)) + 1
    gx, gy = np.meshgrid(lo[0] + spacing * np.arange(cols), hi[1] - spacing * np.arange(rows))
    latlon = geomap.latlon(np.stack([gx, gy], -1))
    if latlon is None:
        raise ValueError("the map has no georeference, so no terrain can be found for it")
    z = np.full(gx.shape, np.nan, np.float32)
    return Terrain(z, spacing, float(lo[0]), float(hi[1])), latlon[..., 0], latlon[..., 1]


def _sample(path: str | Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear DEM values at WGS84 points (NaN outside it or on its no-data cells)."""
    import rasterio  # optional dependency
    from pyproj import Transformer
    from rasterio.windows import Window

    out = np.full(lat.shape, np.nan, np.float32)
    with rasterio.open(path) as ds:
        x, y = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True).transform(lon, lat)
        inv = ~ds.transform
        x, y = np.asarray(x), np.asarray(y)
        # from pixel corners to pixel centres
        col, row = inv.a * x + inv.b * y + inv.c - 0.5, inv.d * x + inv.e * y + inv.f - 0.5
        inside = (col >= -0.5) & (col <= ds.width - 0.5) & (row >= -0.5) & (row <= ds.height - 0.5)
        if not inside.any():
            return out
        c0 = max(int(np.floor(col[inside].min())) - 1, 0)
        r0 = max(int(np.floor(row[inside].min())) - 1, 0)
        c1 = min(int(np.ceil(col[inside].max())) + 2, ds.width)
        r1 = min(int(np.ceil(row[inside].max())) + 2, ds.height)
        band = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), masked=True)
    z = band.astype(np.float32).filled(np.nan)
    at = [row[inside] - r0, col[inside] - c0]
    out[inside] = ndimage.map_coordinates(z, at, order=1, mode="nearest")
    return out


def _fill_nearest(z: np.ndarray) -> np.ndarray:
    missing = np.isnan(z)
    if not missing.any():
        return z
    if missing.all():
        return np.zeros_like(z)
    index = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
    return z[tuple(index)]
