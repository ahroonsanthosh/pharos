"""Georeferenced reference imagery (the "satellite map")."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

import cv2
import numpy as np

EARTH_RADIUS_M = 6_378_137.0


@dataclass
class GeoMap:
    """A north-up orthoimage with a simple affine georeference.

    Pixel ``(u, v)`` (centre of column ``u``, row ``v``) sits at world
    ``x = origin_x + u * gsd``, ``y = origin_y - v * gsd``.

    World coordinates are either a projected CRS in metres (``crs``, e.g. a
    UTM zone for a GeoTIFF orthophoto) or a local frame whose (0, 0) is at
    ``geo_origin`` = (lat, lon), as the simulator uses. Either way positions
    can be reported as WGS84 latitude/longitude.
    """

    image: np.ndarray
    gsd: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    geo_origin: tuple[float, float] | None = None
    crs: str | None = None
    _pyramid: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) of pixel centres, metres."""
        h, w = self.shape
        return (
            self.origin_x,
            self.origin_y - (h - 1) * self.gsd,
            self.origin_x + (w - 1) * self.gsd,
            self.origin_y,
        )

    @cached_property
    def gray(self) -> np.ndarray:
        if self.image.ndim == 2:
            return self.image
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def level(self, n: int) -> tuple[np.ndarray, float]:
        """Gray image downsampled by 2**n, and its ground sample distance."""
        if n not in self._pyramid:
            img = self.gray
            for _ in range(n):
                img = cv2.pyrDown(img)
            self._pyramid[n] = img
        return self._pyramid[n], self.gsd * 2**n

    def world_to_px(self, xy: np.ndarray, level: int = 0) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        g = self.gsd * 2**level
        return np.stack([(xy[..., 0] - self.origin_x) / g, (self.origin_y - xy[..., 1]) / g], -1)

    def px_to_world(self, uv: np.ndarray, level: int = 0) -> np.ndarray:
        uv = np.asarray(uv, dtype=float)
        g = self.gsd * 2**level
        return np.stack([self.origin_x + uv[..., 0] * g, self.origin_y - uv[..., 1] * g], -1)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        x0, y0, x1, y1 = self.extent
        return x0 + margin <= x <= x1 - margin and y0 + margin <= y <= y1 - margin

    def to_latlon(self, x: float, y: float) -> tuple[float, float] | None:
        if self.crs is not None:
            lon, lat = _to_wgs84(self.crs).transform(x, y)
            return float(lat), float(lon)
        if self.geo_origin is None:
            return None
        lat0, lon0 = self.geo_origin
        lat = lat0 + np.degrees(y / EARTH_RADIUS_M)
        lon = lon0 + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
        return float(lat), float(lon)

    def from_latlon(self, lat: float, lon: float) -> tuple[float, float] | None:
        """World coordinates of a WGS84 position (inverse of :meth:`to_latlon`)."""
        if self.crs is not None:
            x, y = _to_wgs84(self.crs).transform(lon, lat, direction="INVERSE")
            return float(x), float(y)
        if self.geo_origin is None:
            return None
        lat0, lon0 = self.geo_origin
        y = np.radians(lat - lat0) * EARTH_RADIUS_M
        x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
        return float(x), float(y)

    def resampled(self, gsd: float) -> GeoMap:
        """The same map at a coarser ground sample distance (anti-aliased)."""
        f = gsd / self.gsd
        if f <= 1.0:
            return self
        h, w = self.shape
        size = (int(w / f), int(h / f))
        # Destination pixel u samples source u * f + (f - 1) / 2, the centre of its footprint.
        M = np.array([[f, 0.0, (f - 1) / 2], [0.0, f, (f - 1) / 2]])
        blurred = cv2.GaussianBlur(self.image, (0, 0), 0.5 * f)
        image = cv2.warpAffine(blurred, M, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        half = (f - 1) / 2 * self.gsd
        return GeoMap(image, gsd, self.origin_x + half, self.origin_y - half, self.geo_origin,
                      self.crs)  # fmt: skip

    def save(self, path: str | Path) -> None:
        """Write ``<path>.png`` and a ``<path>.json`` georeference sidecar."""
        path = Path(path).with_suffix("")
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path.with_suffix(".png")), self.image)
        meta = {
            "image": path.with_suffix(".png").name,
            "gsd": self.gsd,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "geo_origin": self.geo_origin,
            "crs": self.crs,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> GeoMap:
        """Load from a ``.json`` sidecar (or the image next to it)."""
        meta_path = Path(path).with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        image = cv2.imread(str(meta_path.parent / meta["image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(meta_path.parent / meta["image"])
        geo = meta.get("geo_origin")
        return cls(
            image=image,
            gsd=float(meta["gsd"]),
            origin_x=float(meta.get("origin_x", 0.0)),
            origin_y=float(meta.get("origin_y", 0.0)),
            geo_origin=tuple(geo) if geo else None,
            crs=meta.get("crs"),
        )

    @classmethod
    def from_geotiff(cls, path: str | Path) -> GeoMap:
        """Load a north-up orthophoto GeoTIFF in a projected CRS (needs ``pharos[geo]``).

        Reproject other inputs first, e.g. ``gdalwarp -t_srs EPSG:32632 in.tif out.tif``.
        """
        import rasterio  # optional dependency

        with rasterio.open(path) as ds:
            if ds.crs is None or not ds.crs.is_projected:
                raise ValueError(f"{path}: need a projected CRS in metres (e.g. UTM)")
            t = ds.transform
            if abs(t.b) > 1e-9 or abs(t.d) > 1e-9 or not np.isclose(t.a, -t.e, rtol=1e-6):
                raise ValueError(f"{path}: need a north-up raster with square pixels")
            bands = ds.read(indexes=list(range(1, min(ds.count, 3) + 1)))
            crs = ds.crs.to_string()
        image = _to_uint8(np.moveaxis(bands, 0, -1))
        image = image[..., 0] if image.shape[-1] != 3 else image[..., ::-1].copy()  # RGB -> BGR
        return cls(
            image=np.ascontiguousarray(image),
            gsd=float(t.a),
            origin_x=float(t.c + t.a / 2),
            origin_y=float(t.f + t.e / 2),
            crs=crs,
        )


@lru_cache(maxsize=4)
def _to_wgs84(crs: str):
    from pyproj import Transformer  # optional dependency

    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Stretch 16-bit or float imagery (e.g. Sentinel-2) to 8 bits; pass uint8 through."""
    if img.dtype == np.uint8:
        return img
    lo, hi = np.percentile(img, [1, 99])
    return (np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
