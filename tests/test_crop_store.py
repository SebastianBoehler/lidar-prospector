import numpy as np

from lidarprospector.crop_store import CropStore


def terrain(seed, size=400, busy=False):
    rng = np.random.default_rng(seed)
    z = np.full((size, size), 50.0, np.float32)
    if busy:  # a lot of relief in one corner
        y, x = np.mgrid[:size, :size]
        z += 3 * np.sin(x / 5) * np.cos(y / 6) * ((x < 150) & (y < 150))
    return z + 0.01 * rng.standard_normal((size, size)).astype(np.float32)


def test_store_builds_once_and_samples_crops_of_the_right_shape(tmp_path):
    calls = []

    def loader(seed, **kw):
        def load():
            calls.append(seed)
            return terrain(seed, **kw)

        return load

    items = [("a", loader(1)), ("b", loader(2, busy=True))]
    store = CropStore.build(items, tmp_path, crop=64)
    again = CropStore.build(items, tmp_path, crop=64)  # reuses the cache: loaders not called again
    assert calls == [1, 2] and len(store) == len(again) == 2
    x = store.sample(10, 64, np.random.default_rng(0))
    assert x.shape == (10, 2, 64, 64) and x.dtype == np.float32 and np.isfinite(x).all()


def test_energy_weighting_favours_structured_terrain(tmp_path):
    store = CropStore.build(
        [("flat", lambda: terrain(1)), ("busy", lambda: terrain(2, busy=True))], tmp_path, crop=64
    )
    rng = np.random.default_rng(0)
    energy = [np.abs(store.sample(200, 64, rng)[:, 0]).mean() for _ in range(3)]
    flat_only = np.abs(
        CropStore.build([("flat", lambda: terrain(1))], tmp_path / "f", crop=64).sample(200, 64, rng)[:, 0]
    ).mean()
    assert np.mean(energy) > 2 * flat_only  # the mixture pulls in far more relief than flat ground alone


def test_wrong_crop_size_is_rejected(tmp_path):
    import pytest

    store = CropStore.build([("a", lambda: terrain(1))], tmp_path, crop=64)
    with pytest.raises(ValueError, match="crop=64"):
        store.sample(2, 32, np.random.default_rng(0))
