import numpy as np
import rasterio
from rasterio.transform import from_origin

from lidarprospector.sources.scotland import ScotlandSRSP, quarter_code, tile_code, tile_origin


def test_grid_codes_round_trip():
    assert tile_origin("NH34") == (230_000, 840_000)  # checked against the real file's bounds
    assert tile_origin("SU14") == (410_000, 140_000)  # Stonehenge
    for code in ("NH34", "NT25", "NN67", "HY20", "NS56", "SU14", "TQ38"):
        e, n = tile_origin(code)
        assert tile_code(e + 5_000, n + 9_999) == code


def test_quarter_tiles():
    assert tile_origin("NS16NE") == (215_000, 665_000)
    assert quarter_code(215_100, 665_100) == "NS16NE" and quarter_code(210_100, 660_100) == "NS16SW"


def write_tile(path, e, n, value):
    z = np.full((100, 100), value, np.float32)
    profile = {"driver": "GTiff", "height": 100, "width": 100, "count": 1, "dtype": "float32"}
    profile["nodata"] = -9999
    with rasterio.open(path, "w", crs="EPSG:27700", transform=from_origin(e, n + 100, 1, 1), **profile) as d:
        d.write(z, 1)


def test_mosaic_across_two_tiles_and_missing_tile(tmp_path):
    # a small file stands in for a 10 km tile (the lookup is tested, not the size)
    write_tile(tmp_path / "a.tif", 230_000, 840_000, 5.0)
    src = ScotlandSRSP(index={"NH34": str(tmp_path / "a.tif")})
    dem = src.fetch((230_050, 840_020, 230_100, 840_060))
    assert dem.z.shape == (40, 50) and np.allclose(dem.z, 5.0)
    gap = src.fetch((230_000, 850_000, 230_050, 850_050))  # the tile north of it is not in the index
    assert np.isnan(gap.z).all()
    assert dem.licence.startswith("Open Government Licence")
