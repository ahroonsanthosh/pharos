"""Reference maps kept on disk for the service."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import cv2

from pharos.geomap import GeoMap


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not cleaned:
        raise ValueError("empty name")
    return cleaned


class MapStore:
    """Maps saved as PNG + JSON georeference under ``root``.

    Imagery finer than ``match_gsd`` is resampled on import: matching runs at
    about 1 m per pixel, and keeping 10 cm imagery would cost 100x the memory
    and search time for no gain in accuracy.
    """

    def __init__(self, root: Path, match_gsd: float = 1.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.match_gsd = match_gsd
        self._cache: dict[str, GeoMap] = {}
        self._lock = threading.Lock()

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def get(self, name: str) -> GeoMap:
        with self._lock:
            if name not in self._cache:
                path = self.root / f"{safe_name(name)}.json"
                if not path.exists():
                    raise KeyError(f"no map named {name!r}")
                self._cache[name] = GeoMap.load(path)
            return self._cache[name]

    def add(self, name: str, geomap: GeoMap) -> str:
        name = safe_name(name)
        geomap = geomap.resampled(self.match_gsd)
        geomap.save(self.root / name)
        with self._lock:
            self._cache[name] = geomap
        return name

    def add_file(self, name: str, path: Path) -> str:
        """Import a GeoTIFF (``.tif``) or a PNG/JPEG with a ``.json`` sidecar next to it."""
        suffix = path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            geomap = GeoMap.from_geotiff(path)
        elif suffix == ".json":
            geomap = GeoMap.load(path)
        else:
            raise ValueError("upload a GeoTIFF (.tif) or a map .json with its image")
        return self.add(name, geomap)

    def info(self, name: str) -> dict:
        gm = self.get(name)
        h, w = gm.shape
        x0, y0, x1, y1 = gm.extent
        return {
            "name": name,
            "width": w,
            "height": h,
            "gsd": gm.gsd,
            "extent": [x0, y0, x1, y1],
            "size_km": [round((x1 - x0) / 1000, 2), round((y1 - y0) / 1000, 2)],
            "crs": gm.crs,
        }

    def preview(self, name: str, max_px: int = 2048) -> bytes:
        """JPEG of the whole map, at most ``max_px`` on its longer side."""
        gm = self.get(name)
        scale = min(1.0, max_px / max(gm.shape))
        img = gm.image
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
