"""Terrain models: lookup, resampling DEM files onto a map, Copernicus tiles."""

import io
import urllib.error

import numpy as np
import pytest

from pharos.geomap import GeoMap
from pharos.terrain import Terrain, copernicus_tile


def slope(lat, lon):
    """A tilted plane: 1000 m per degree of latitude, 500 per degree of longitude."""
    return 300.0 + 1000.0 * (lat - 47.0) + 500.0 * (lon - 15.9)


def write_dem(path, west, north, n=400, step=1 / 3600):
    """A north-up EPSG:4326 GeoTIFF of ``slope``, pixel centres at whole steps."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    lat = north - step * np.arange(n)
    lon = west + step * np.arange(n)
    z = slope(lat[:, None], lon[None, :]).astype(np.float32)
    transform = from_origin(west - step / 2, north + step / 2, step, step)
    profile = dict(driver="GTiff", width=n, height=n, count=1, dtype="float32")
    with rasterio.open(path, "w", crs="EPSG:4326", transform=transform, **profile) as ds:
        ds.write(z, 1)
    return path


def small_map():
    """600 by 400 m around 47.01 N, 15.91 E: inside one Copernicus tile."""
    return GeoMap(np.zeros((200, 300, 3), np.uint8), 2.0, geo_origin=(47.01, 15.91))


def test_lookup_is_bilinear_and_holds_the_edge_value():
    t = Terrain(np.array([[0, 10], [20, 30]], np.float32), 10.0, 0.0, 10.0)
    xy = np.array([[0, 10], [5, 5], [10, 0], [50, -50], [np.nan, 0]])
    assert t.at(xy)[:4] == pytest.approx([0, 15, 30, 30])
    assert np.isnan(t.at(xy)[4])


def test_dem_file_is_resampled_onto_the_map(tmp_path):
    pytest.importorskip("pyproj")
    dem = write_dem(tmp_path / "dem.tif", west=15.89, north=47.03)
    gm = small_map()
    terrain = Terrain.from_dem([dem], gm, margin_m=100.0)
    for x, y in [(0.0, 0.0), (300.0, -200.0), (550.0, 100.0)]:
        lat, lon = gm.to_latlon(x, y)
        assert terrain.at(np.array([x, y])) == pytest.approx(slope(lat, lon), abs=0.5)


def test_map_keeps_its_terrain_through_save_and_resampling(tmp_path):
    gm = small_map()
    gm.terrain = Terrain(np.arange(16, dtype=np.float32).reshape(4, 4), 100.0, -50.0, 350.0)
    gm.save(tmp_path / "area")
    loaded = GeoMap.load(tmp_path / "area.json").resampled(4.0)
    assert np.array_equal(loaded.terrain.elevation, gm.terrain.elevation)


def test_copernicus_tiles_are_fetched_once_and_open_sea_is_zero(tmp_path, monkeypatch):
    tile = write_dem(tmp_path / "source.tif", west=15.0, north=48.0, n=3600)
    fetched = []

    def urlopen(url, timeout):
        fetched.append(url.rsplit("/", 1)[1])
        if "N47_00_E015" not in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return io.BytesIO(tile.read_bytes())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    gm = small_map()
    terrain = Terrain.copernicus(gm, margin_m=100.0, cache=tmp_path)
    lat, lon = gm.to_latlon(100.0, 50.0)
    assert terrain.at(np.array([100.0, 50.0])) == pytest.approx(slope(lat, lon), abs=0.5)
    Terrain.copernicus(gm, margin_m=100.0, cache=tmp_path)  # from the cache
    assert fetched == ["Copernicus_DSM_COG_10_N47_00_E015_00_DEM.tif"]

    sea = copernicus_tile(-34, -1, tmp_path / "copernicus")
    assert sea is None and fetched[-1] == "Copernicus_DSM_COG_10_S34_00_W001_00_DEM.tif"
    copernicus_tile(-34, -1, tmp_path / "copernicus")  # remembered as sea
    assert len(fetched) == 2
