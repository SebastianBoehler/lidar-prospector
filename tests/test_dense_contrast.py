import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lidarprospector.dense_contrast import anchor_loss, apply_g, invert_g, pretrain  # noqa: E402


def test_invert_undoes_every_dihedral_member_exactly():
    x = torch.randn(16, 2, 32, 32)
    g = torch.arange(16) % 8
    assert torch.equal(invert_g(apply_g(x, g), g), x)


def test_aligned_views_give_a_lower_loss_than_misaligned_ones():
    """If positions line up the loss must be low for matching features and high once one view is rolled."""
    torch.manual_seed(0)
    f = torch.randn(8, 16, 64, 64)
    proj = torch.nn.Identity()
    aligned = anchor_loss(f, f.clone(), proj, per_crop=16).item()
    shifted = anchor_loss(f, torch.roll(f, shifts=(8, 8), dims=(2, 3)), proj, per_crop=16).item()
    assert aligned < 0.5 * shifted


def test_pretraining_runs_and_reduces_the_loss_on_structured_terrain():
    y, x = np.mgrid[:300, :300].astype(float)
    terrain = 50 + 3 * np.sin(x / 11) * np.cos(y / 9) + 2 * np.sin((x + y) / 17)
    logs = []

    def record(s, **_):
        logs.append(s)

    pretrain([terrain.astype("float32")], steps=60, batch=8, crop=64, per_crop=8, device="cpu", log=record)
    first = float(logs[0].split("loss")[1].split()[0])
    last = float(logs[-1].split("loss")[1].split()[0])
    assert last < first
