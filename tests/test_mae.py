import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lidarprospector.mae import block_mask, pretrain  # noqa: E402


def test_block_mask_hides_whole_aligned_blocks_at_about_the_requested_ratio():
    m = block_mask(64, 128, 0.5, "cpu", block=16)
    assert m.shape == (64, 1, 128, 128)
    assert 0.4 < m.mean().item() < 0.6
    tiles = m.reshape(64, 8, 16, 8, 16)  # every 16 x 16 block is all hidden or all visible
    assert (tiles.amax(dim=(2, 4)) == tiles.amin(dim=(2, 4))).all()


def test_pretraining_reduces_the_masked_reconstruction_error_on_structured_terrain():
    y, x = np.mgrid[:300, :300].astype(float)
    terrain = 50 + 3 * np.sin(x / 11) * np.cos(y / 9) + 2 * np.sin((x + y) / 17)
    logs = []
    pretrain(
        [terrain.astype("float32")],
        steps=60,
        batch=8,
        crop=64,
        device="cpu",
        log=lambda s, **_: logs.append(s),
    )
    first = float(logs[0].split("masked-MSE")[1].split()[0])
    last = float(logs[-1].split("masked-MSE")[1].split()[0])
    assert last < 0.7 * first
