"""Mound / barrow detection from a dense probability map, and the trained U-Net as a tile function."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .embed import relief_channels
from .features_torch import pick_device
from .peaks import peaks_from_prob  # noqa: F401  (re-exported)
from .unet import UNet


@torch.no_grad()
def predict_dense(model, chan: np.ndarray, device: str, block: int = 1024, pad: int = 32) -> np.ndarray:
    """Run a fully convolutional model over an image in blocks (bounded memory), blending nothing:
    each block is predicted with `pad` pixels of context that are then discarded."""
    model.eval()
    _, h, w = chan.shape
    ph, pw = (-h) % block, (-w) % block
    x = F.pad(torch.from_numpy(chan)[None], (pad, pad + pw, pad, pad + ph), mode="reflect").to(device)
    out = torch.zeros(h + ph, w + pw)
    for r in range(0, h + ph, block):
        for c in range(0, w + pw, block):
            tile = x[:, :, r : r + block + 2 * pad, c : c + block + 2 * pad]
            out[r : r + block, c : c + block] = torch.sigmoid(model(tile))[0, pad:-pad, pad:-pad].cpu()
    return out[:h, :w].numpy()


def unet_mound_fn(weights: Path, device: str = "auto") -> Callable[[np.ndarray], np.ndarray]:
    """dtm -> barrow probability map, using weights saved by `train_unet`."""
    device = pick_device(device)
    model = UNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    return lambda dtm: predict_dense(model, relief_channels(dtm), device)
