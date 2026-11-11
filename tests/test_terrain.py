import numpy as np

from lidarprospector.features import RELIEF_SIGMAS, feature_stack, fill_nodata
from lidarprospector.terrain import ground_dtm, hillshade, local_relief


def bump(size=200, height=1.0, sigma=5.0):
    y, x = np.mgrid[:size, :size]
    r2 = (x - size / 2) ** 2 + (y - size / 2) ** 2
    return height * np.exp(-r2 / (2 * sigma**2))


def test_local_relief_marks_mound_positive_and_flat_ground_zero():
    rel = local_relief(bump())
    assert rel[100, 100] > 0.5
    assert abs(rel[5, 5]) < 0.05


def test_local_relief_marks_pit_negative():
    assert local_relief(-bump())[100, 100] < -0.5


def test_hillshade_of_flat_ground_is_uniform():
    shade = hillshade(np.zeros((50, 50)))
    assert np.allclose(shade, np.sin(np.radians(40.0)))


def test_ground_dtm_uses_only_ground_returns_and_fills_small_gaps():
    xs, ys = np.meshgrid(np.arange(0, 50, 0.5), np.arange(0, 50, 0.5))
    x, y = xs.ravel(), ys.ravel()
    z = np.full_like(x, 10.0)
    cls = np.full(x.shape, 2)
    cls[::7] = 5  # vegetation returns, must be ignored
    z[cls == 5] = 99.0
    dtm, (x0, y0, cell) = ground_dtm(x, y, z, cls)
    assert np.nanmax(dtm) == 10.0
    assert (x0, y0, cell) == (0.0, 0.0, 1.0)


def test_fill_nodata_leaves_no_nan():
    a = np.ones((20, 20))
    a[5:8, 5:8] = np.nan
    assert not np.isnan(fill_nodata(a)).any()


def test_feature_stack_shape_names_and_nodata_propagation():
    dtm = bump(120)
    dtm[:10, :10] = np.nan
    feats, names = feature_stack(dtm)
    assert feats.shape == (120, 120, len(names))
    assert len(names) == len(RELIEF_SIGMAS) + 6 + 3
    assert np.isnan(feats[0, 0]).all()
    assert np.isfinite(feats[60, 60]).all()
