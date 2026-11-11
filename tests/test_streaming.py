import numpy as np
import pytest
import rasterio
from rasterio.shutil import copy as rcopy
from rasterio.transform import from_origin

from lidarprospector.rangeserver import serve
from lidarprospector.sources import GeoTiffSource


@pytest.fixture(scope="module")
def cog(tmp_path_factory):
    """A real cloud-optimised GeoTIFF (512 px tiles, overviews) of noisy terrain, served over HTTP."""
    d = tmp_path_factory.mktemp("cog")
    rng = np.random.default_rng(0)
    z = (50 + np.cumsum(rng.standard_normal((2048, 2048)), axis=0) * 0.05).astype("float32")
    z += rng.standard_normal(z.shape).astype("float32") * 0.2  # incompressible detail: bytes track pixels
    profile = {"driver": "GTiff", "height": 2048, "width": 2048, "count": 1, "dtype": "float32",
               "crs": "EPSG:27700", "transform": from_origin(1000, 5000, 1, 1)}  # fmt: skip
    with rasterio.MemoryFile() as mf:
        with mf.open(**profile) as ds:
            ds.write(z, 1)
        with mf.open() as src:
            rcopy(src, d / "t.tif", driver="COG", BLOCKSIZE="512", COMPRESS="DEFLATE", PREDICTOR="3")
    server, meter, base = serve(d / "t.tif")
    yield d / "t.tif", z, meter, base
    server.shutdown()
    server.server_close()


def source(cog, tag):
    path, _, meter, base = cog
    meter.reset()
    return GeoTiffSource(f"{base}/{tag}.tif", "l", "a"), meter, path.stat().st_size


def test_streamed_window_equals_the_local_read_and_costs_less_than_the_file(cog):
    src, meter, size = source(cog, "window")
    dem = src.fetch((1500, 3000, 2100, 3600))  # 600 x 600 m at native 1 m
    src.close()
    assert np.array_equal(dem.z, cog[1][1400:2000, 500:1100])
    assert meter.bytes < 0.35 * size  # only the touched 512 px tiles were requested


def test_a_coarse_look_uses_overviews_and_costs_a_few_percent(cog):
    src, meter, size = source(cog, "coarse")
    dem = src.fetch((1000, 2952, 3048, 5000), cell=8.0)  # the whole file at 8 m
    src.close()
    assert dem.z.shape == (256, 256)
    assert meter.bytes < 0.06 * size  # an overview read, not a full read averaged down


def test_overlapping_windows_do_not_download_the_overlap_twice(cog):
    src, meter, _ = source(cog, "overlap")
    src.fetch((1000, 3000, 1800, 3800))
    first = meter.bytes
    src.fetch((1400, 3000, 2200, 3800))  # shares half its pixels with the first window
    second = meter.bytes - first
    src.close()
    assert second < 0.75 * first  # the shared tiles came from the decoded-block cache
