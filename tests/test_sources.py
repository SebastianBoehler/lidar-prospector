import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from lidarprospector.sources import GeoTiffSource, source_for
from lidarprospector.sources.base import require_metric, utm_epsg


def write_tif(path, z, transform, crs):
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform) as ds:  # fmt: skip
        ds.write(z.astype("float32"), 1)


def test_projected_geotiff_window_is_read_exactly(tmp_path):
    z = np.arange(400 * 400, dtype="float32").reshape(400, 400)
    write_tif(tmp_path / "dem.tif", z, from_origin(1000, 5000, 1, 1), "EPSG:27700")
    src = GeoTiffSource(str(tmp_path / "dem.tif"), "licence", "attribution")
    dem = src.fetch((1100, 4700, 1200, 4800))  # x 1100..1200, y 4700..4800 -> rows 200..300, cols 100..200
    assert dem.epsg == 27700 and dem.cell == 1.0
    assert np.array_equal(dem.z, z[200:300, 100:200])
    assert (dem.x0, dem.y1) == (1100, 4800)
    assert dem.licence == "licence"


def test_outside_the_file_is_nodata_not_an_error(tmp_path):
    write_tif(tmp_path / "dem.tif", np.ones((50, 50)), from_origin(0, 50, 1, 1), "EPSG:27700")
    dem = GeoTiffSource(str(tmp_path / "dem.tif"), "l", "a").fetch((-20, 0, 20, 50))
    assert np.isnan(dem.z[:, :20]).all() and np.isfinite(dem.z[:, 20:]).all()


def test_geographic_raster_is_reprojected_to_metric_and_georeferenced(tmp_path):
    """Elevation = a plane in lon/lat; after reprojection each pixel must hold the plane's value at its
    own lon/lat, which only holds if the geotransform and reprojection are right."""
    lon0, lat0, step = -2.0, 51.5, 1 / 3600  # 1 arc-second
    n = 400
    lon = lon0 + (np.arange(n) + 0.5) * step
    lat = lat0 - (np.arange(n) + 0.5) * step
    z = 100 + 2000 * (lon[None, :] - lon0) + 3000 * (lat0 - lat[:, None])
    write_tif(tmp_path / "geo.tif", z, from_origin(lon0, lat0, step, step), "EPSG:4326")
    src = GeoTiffSource(str(tmp_path / "geo.tif"), "l", "a")
    assert src.epsg == utm_epsg(-2.0, 51.4) == 32630
    cx, cy = Transformer.from_crs(4326, 32630, always_xy=True).transform(lon0 + 0.05, lat0 - 0.05)
    dem = src.fetch((cx - 600, cy - 600, cx + 600, cy + 600), cell=30)
    r, c = 20, 20
    plon, plat = dem.to_wgs84(r + 0.5, c + 0.5)
    expect = 100 + 2000 * (plon - lon0) + 3000 * (lat0 - plat)
    assert abs(dem.z[r, c] - expect) < 0.5


def test_metric_crs_is_required():
    with pytest.raises(ValueError, match="metric"):
        require_metric(4326)
    assert require_metric(27700) == 27700


def test_registry_reports_what_is_registered_when_nothing_covers_a_place():
    assert source_for(-1.83, 51.18).name.startswith("England")
    with pytest.raises(LookupError, match="no registered terrain source"):
        source_for(-91.4, 32.6)


def test_reducing_resolution_averages_instead_of_aliasing(tmp_path):
    """A 0.25 m raster read at 1 m must give the mean of each 4x4 block, not a sampled value."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((400, 400)).astype("float32")  # white noise: aliasing would show immediately
    write_tif(tmp_path / "fine.tif", z, from_origin(1000, 5100, 0.25, 0.25), "EPSG:27700")
    src = GeoTiffSource(str(tmp_path / "fine.tif"), "l", "a")
    dem = src.fetch((1000, 5000, 1100, 5100), cell=1.0)
    expect = z.reshape(100, 4, 100, 4).mean(axis=(1, 3))
    assert np.abs(dem.z - expect).max() < 1e-4


def test_presets_carry_licence_and_citation():
    from lidarprospector.sources.presets import PRESETS

    for key, p in PRESETS.items():
        assert p["licence"] and p["attribution"] and p["url"].startswith("https://"), key
