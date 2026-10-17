import numpy as np
import pytest

from pharos.geomap import GeoMap


def make_map():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 80, 3), dtype=np.uint8)
    return GeoMap(img, gsd=2.0, origin_x=100.0, origin_y=500.0, geo_origin=(46.0, 7.0))


def test_pixel_world_round_trip():
    gm = make_map()
    uv = np.array([[0.0, 0.0], [10.5, 3.25], [79.0, 63.0]])
    np.testing.assert_allclose(gm.world_to_px(gm.px_to_world(uv)), uv)
    np.testing.assert_allclose(gm.px_to_world([1.0, 1.0]), [102.0, 498.0])


def test_levels_halve_resolution():
    gm = make_map()
    img, gsd = gm.level(1)
    assert img.shape == (32, 40)
    assert gsd == 4.0
    np.testing.assert_allclose(gm.world_to_px([108.0, 492.0], level=1), [2.0, 2.0])


def test_extent_and_contains():
    gm = make_map()
    assert gm.extent == (100.0, 500.0 - 63 * 2, 100.0 + 79 * 2, 500.0)
    assert gm.contains(150.0, 450.0)
    assert not gm.contains(50.0, 450.0)


def test_latlon_near_origin():
    gm = make_map()
    lat, lon = gm.to_latlon(0.0, 1000.0)
    assert np.isclose(lon, 7.0)
    assert np.isclose(lat - 46.0, 1000.0 / 6378137.0 * 180 / np.pi)


def test_save_load_round_trip(tmp_path):
    gm = make_map()
    gm.save(tmp_path / "map")
    loaded = GeoMap.load(tmp_path / "map.json")
    np.testing.assert_array_equal(loaded.image, gm.image)
    assert (loaded.gsd, loaded.origin_x, loaded.origin_y) == (2.0, 100.0, 500.0)
    assert loaded.geo_origin == (46.0, 7.0)


def test_geotiff_in_utm(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    rgb = np.zeros((3, 20, 30), np.uint8)
    rgb[0] = 200  # red
    path = tmp_path / "ortho.tif"
    profile = {
        "driver": "GTiff",
        "width": 30,
        "height": 20,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:32632",
        "transform": from_origin(500000.0, 5200000.0, 0.5, 0.5),
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(rgb)

    gm = GeoMap.from_geotiff(path)
    assert gm.shape == (20, 30) and gm.gsd == 0.5
    assert tuple(gm.image[0, 0]) == (0, 0, 200)  # BGR
    np.testing.assert_allclose(gm.px_to_world([0, 0]), [500000.25, 5199999.75])
    lat, lon = gm.to_latlon(500000.0, 5200000.0)
    assert np.isclose(lon, 9.0, atol=1e-6) and 46.9 < lat < 47.0  # UTM 32N central meridian
