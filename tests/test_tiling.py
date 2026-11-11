import numpy as np
import pytest

from lidarprospector.pipeline import ScanConfig, scan
from lidarprospector.runtime import Hardware, MemoryGate, Plan
from lidarprospector.shapes import Line
from lidarprospector.sources.base import Dem
from lidarprospector.terrain import local_relief
from lidarprospector.tiling import (
    TileGrid,
    downsample,
    merge_line_fragments,
    scales_for,
)


def gauss2d(shape, r0, c0, h, sigma):
    y, x = np.mgrid[: shape[0], : shape[1]]
    return h * np.exp(-((y - r0) ** 2 + (x - c0) ** 2) / (2 * sigma**2))


def landscape(n=1200, seed=0):
    """Gentle noise plus planted features, several straddling the 512 px tile seams."""
    rng = np.random.default_rng(seed)
    z = 50 + 0.02 * rng.standard_normal((n, n))
    y, x = np.mgrid[:n, :n].astype(float)
    z += 1.0 * np.exp(-((x - 500) ** 2 + 0 * y) / (2 * 3.0**2)) * ((y > 100) & (y < 1100))  # vertical bank
    z += 1.0 * np.exp(-((y - 600) ** 2) / (2 * 3.0**2)) * ((x > 100) & (x < 1100))  # horizontal bank
    for r, c in ((300, 300), (400, 510), (700, 514), (900, 900)):
        z -= gauss2d(z.shape, r, c, 1.0, 5.0)  # pits, two on a seam
    return z.astype("float32")


class ArraySource:
    name, licence, attribution, epsg, native_cell = "synthetic", "test", "test", 27700, 1.0

    def __init__(self, z):
        self.z = z

    def covers(self, lon, lat):
        return True

    def fetch(self, bbox, cell=None):
        xmin, ymin, xmax, ymax = (int(v) for v in bbox)
        h, w = ymax - ymin, xmax - xmin
        out = np.full((h, w), np.nan, np.float32)
        r0, c0 = -ymax, xmin  # array row 0 sits at y = 0, rows go south (negative y)
        rs, cs = max(r0, 0), max(c0, 0)
        re, ce = min(r0 + h, self.z.shape[0]), min(c0 + w, self.z.shape[1])
        if re > rs and ce > cs:
            out[rs - r0 : re - r0, cs - c0 : ce - c0] = self.z[rs:re, cs:ce]
        return Dem(out, xmin, ymax, 1.0, 27700, self.name, self.licence, self.attribution, 1.0)


def plan(core, halo):
    hw = Hardware(1, 1 << 40, 1 << 40, "cpu", "test", 0)
    return Plan(hw, 1 << 40, core, halo, 1, 2, 1)


def mound_fn(z):
    return np.nan_to_num(np.clip(local_relief(z) / 1.0, 0, 1))


def run(core, halo):
    z = landscape()
    cfg = ScanConfig(mound_fn=mound_fn, min_mound_prob=0.3, min_extent_m=200)
    src = ArraySource(z)
    return scan(src, (0, -1200, 1200, 0), plan(core, halo), cfg, log=lambda *_: None, gate=MemoryGate(0))


def key_points(res):
    pits = sorted((round(r), round(c)) for r, c, *_ in res.pits)
    mounds = sorted((round(r), round(c)) for r, c, _ in res.mounds)
    return pits, mounds


def line_pixels(res):
    return {(k.kind, int(r), int(c)) for k, _ in res.lines for r, c in k.pixels}


def test_tile_grid_cores_partition_the_area_exactly():
    grid = TileGrid((0, 0, 1000, 700), 1.0, 256, 40)
    covered = np.zeros((700, 1000), int)
    for t in grid:
        covered[t.row0 : t.row0 + t.rows, t.col0 : t.col0 + t.cols] += 1
    assert (covered == 1).all()
    assert len(list(grid)) == len(grid) == 4 * 3


def test_downsample_is_nan_aware_block_mean():
    z = np.array([[1, 3, np.nan, np.nan], [5, 7, np.nan, np.nan]], dtype=np.float32)
    out = downsample(z, 2)
    assert out.shape == (1, 2)
    assert out[0, 0] == 4.0
    assert np.isnan(out[0, 1])


def test_scales_for_selects_pyramid_levels_from_structure_size():
    assert scales_for(5, 60) == (1, 2, 4, 8, 16)
    assert scales_for(100, 400)[0] == 4  # big structures only: the finest levels are skipped
    assert scales_for(1, 8) == (1, 2)


def test_fragments_join_across_a_seam_but_only_for_the_same_kind():
    a = ((0, 0), Line("bank", 0, np.array([[10, c] for c in range(0, 50)])))
    b = ((0, 1), Line("bank", 0, np.array([[10, c] for c in range(50, 100)])))
    c = ((1, 0), Line("ditch", 0, np.array([[10, c] for c in range(50, 100)])))
    merged = merge_line_fragments([a, b, c])
    banks = [m for m in merged if m.kind == "bank"]
    assert len(banks) == 1 and len(banks[0].pixels) == 100
    assert sum(m.kind == "ditch" for m in merged) == 1


@pytest.mark.parametrize("core", [320, 512])
def test_tiled_scan_matches_one_big_tile(core):
    big = run(core=2048, halo=128)  # one tile: the reference answer
    tiled = run(core=core, halo=128)
    assert tiled.tiles > 1 and big.tiles == 1
    assert key_points(tiled) == key_points(big)
    assert len(tiled.lines) == len(big.lines)
    assert line_pixels(tiled) == line_pixels(big)


def test_a_bank_crossing_several_tiles_is_one_long_line():
    res = run(core=320, halo=128)
    lengths = sorted(round(k.extent) for k, _ in res.lines if k.kind == "bank")
    assert lengths and lengths[-1] > 900  # the 1000 px bank survives as a single line
