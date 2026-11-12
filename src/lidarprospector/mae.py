"""Masked terrain modelling: a label-free pretraining objective that asks for dense, local detail.

Random 16 m blocks of the relief are hidden and the network must reconstruct them from context. To do that it
has to learn what terrain structure looks like (mounds, banks, ditches, rings, field edges) at every pixel,
which is the property few-shot prompting needs and the earlier global-contrastive encoder did not provide.
The network is the same U-Net used for mounds (3 input channels: relief x2 and the mask), so its dense
decoder features drop into the prompt pipeline unchanged.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn

from .embed import dihedral, gather, relief_channels
from .promptable import Extractor, blockwise
from .unet import UNet

BLOCK = 16  # default metres hidden as one unit


def make_net() -> UNet:
    net = UNet(in_ch=3)
    net.head = nn.Conv2d(32, 2, 1)  # reconstruct both relief channels
    return net


def block_mask(batch: int, size: int, ratio: float, device, block: int = BLOCK) -> torch.Tensor:
    """(B, 1, size, size) float mask, 1 where hidden, built from block x block pixel blocks."""
    n = size // block
    m = (torch.rand(batch, 1, n, n, device=device) < ratio).float()
    return F.interpolate(m, scale_factor=block, mode="nearest")


def _origins(chans, rng, crop):
    """Per-tile crop origins weighted by local relief energy, so structured terrain is seen often."""
    coords, weights = [], []
    for t, ch in enumerate(chans):
        h, w = ch.shape[1:]
        ys, xs = np.mgrid[0 : h - crop : 16, 0 : w - crop : 16]
        energy = ndimage.uniform_filter(np.abs(ch[0]), crop)[ys + crop // 2, xs + crop // 2]
        coords.append(np.c_[np.full(ys.size, t), ys.ravel(), xs.ravel()])
        weights.append(energy.ravel() + 1e-3)
    coords = np.concatenate(coords)
    p = np.concatenate(weights)
    p = p / p.sum()

    def draw(n):
        half = n // 2
        return np.concatenate(
            [coords[rng.integers(0, len(coords), half)], coords[rng.choice(len(coords), n - half, p=p)]]
        )

    return draw


def corpus_tensors(dtms, rng, crop, device):
    """(T, 2, Hmax, Wmax) relief stack on the device (zero-padded) and a sampler of crop origins that stay
    inside each tile's real extent, weighted toward structured terrain."""
    chans = [relief_channels(d) for d in dtms]
    draw = _origins(chans, rng, crop)
    hmax, wmax = max(c.shape[1] for c in chans), max(c.shape[2] for c in chans)
    padded = [np.pad(c, ((0, 0), (0, hmax - c.shape[1]), (0, wmax - c.shape[2]))) for c in chans]
    return torch.from_numpy(np.stack(padded)).to(device), draw


def pretrain(
    dtms, steps=3000, batch=64, crop=128, ratio=0.5, lr=2e-3, device="mps", seed=0, log=print, block=BLOCK
):
    """Masked reconstruction on every terrain given; no labels. Returns (net, samples_per_second)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tiles, draw = corpus_tensors(dtms, rng, crop, device)
    net = make_net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for step in range(1, steps + 1):
        o = torch.from_numpy(draw(batch)).to(device)
        x = dihedral(gather(tiles, o[:, 0], o[:, 1], o[:, 2], crop))
        x = x * torch.empty(batch, 1, 1, 1, device=device).uniform_(0.8, 1.25)
        m = block_mask(batch, crop, ratio, device, block)
        recon = net.head(net.trunk(torch.cat([x * (1 - m), m], 1)))
        loss = (((recon - x) ** 2) * m).sum() / (2 * m.sum() + 1e-6)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step in (1, steps):
            log(f"step {step:5d}/{steps}  masked-MSE {loss.item():.4f}  {time.time() - t0:.0f}s", flush=True)
    return net, steps * batch / (time.time() - t0)


def mae_extractor(weights: str = "output/mae.pt", device: str | None = None) -> Extractor:
    from .features_torch import pick_device

    device = device or pick_device()
    net = make_net().to(device)
    net.load_state_dict(torch.load(weights, map_location=device))
    net.eval()

    def fn(x):  # nothing hidden at inference: zero mask channel
        return net.trunk(torch.cat([x, torch.zeros_like(x[:, :1])], 1))

    return Extractor(
        "masked-terrain features", lambda dtm: blockwise(fn, relief_channels(dtm), 32, device), 32
    )


def mae_encoder_extractor(weights: str = "output/mae.pt", device: str | None = None) -> Extractor:
    """The masked network's *encoder* feature maps (not the reconstruction decoder), PCA-projected."""
    from .features_torch import pick_device
    from .promptable import multiscale_taps, pca_extractor

    device = device or pick_device()
    net = make_net().to(device)
    net.load_state_dict(torch.load(weights, map_location=device))
    net.eval()
    raw = multiscale_taps(net.body, taps=(0, 2, 4, 5), extra_zero_channels=1)
    return pca_extractor("masked-terrain ENCODER features (PCA 32)", raw, 32 + 64 + 128 + 192, device)
