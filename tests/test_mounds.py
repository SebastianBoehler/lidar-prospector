import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lidarprospector.mounds import predict_dense  # noqa: E402


class Identity(torch.nn.Module):
    """A 'network' whose logit is the first channel, so the output must equal the input."""

    def forward(self, x):
        return x[:, 0] * 8.0


@pytest.mark.parametrize("shape", [(300, 200), (1024, 1024), (1500, 700)])
def test_block_inference_reproduces_a_pointwise_model_at_any_size(shape):
    chan = np.random.default_rng(0).uniform(-0.2, 0.2, (2, *shape)).astype("float32")
    out = predict_dense(Identity(), chan, "cpu", block=1024, pad=32)
    assert out.shape == shape
    assert np.allclose(out, 1 / (1 + np.exp(-chan[0] * 8.0)), atol=1e-5)


def test_images_smaller_than_the_context_are_rejected():
    with pytest.raises(ValueError, match="context"):
        predict_dense(Identity(), np.zeros((2, 20, 20), "float32"), "cpu")
