import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lidarprospector.features import feature_stack  # noqa: E402
from lidarprospector.features_torch import feature_stack_torch  # noqa: E402


def terrain(n=760, seed=1):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:n, :n].astype(float)
    z = 120 + 0.01 * y + 1.5 * np.exp(-((x - 380) ** 2 + (y - 380) ** 2) / (2 * 9.0**2))
    z -= 0.8 * np.exp(-((x - 200) ** 2 + (y - 500) ** 2) / (2 * 6.0**2))
    return (z + 0.02 * rng.standard_normal((n, n))).astype("float32")


def test_torch_features_match_scipy_in_the_interior():
    dtm = terrain()
    ref, names = feature_stack(dtm)
    got, names_t = feature_stack_torch(dtm, device="cpu")
    assert names == names_t
    m = 305  # beyond the widest filter radius, where edge handling cannot matter
    for i, name in enumerate(names):
        a, b = ref[m:-m, m:-m, i], got[m:-m, m:-m, i]
        scale = max(float(np.abs(a).max()), 1e-6)
        assert np.abs(a - b).max() / scale < 1e-2, name


def test_second_derivative_kernel_ignores_the_mean_elevation():
    """A constant 150 m surface must have exactly zero curvature (truncation must not leak the mean)."""
    flat = np.full((700, 700), 150.0, dtype=np.float32)
    feats, names = feature_stack_torch(flat, device="cpu")
    for i, name in enumerate(names):
        assert np.abs(feats[..., i]).max() < 1e-3, name


def test_too_small_a_tile_is_rejected_clearly():
    with pytest.raises(ValueError, match="filter radius"):
        feature_stack_torch(np.zeros((100, 100), dtype=np.float32), device="cpu")
