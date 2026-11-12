"""Dense contrastive pretraining: the feature at a place must match the same place seen through a different
rotation/flip and amplitude, and differ from other places. Unlike masked reconstruction it rewards features
that *separate local shapes*, and it makes them rotation-invariant, which is what shape prompts need.

Both views come from one crop, so positions align exactly once each view's dihedral transform is undone.
No labels. The network is the mound U-Net (2 input channels), so its decoder features load through
`promptable.unet_extractor` unchanged.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .mae import batch_sampler
from .unet import UNet

MARGIN, STRIDE = 16, 8  # anchors on a coarse grid inside the crop, so negatives are never near-duplicates


def apply_g(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Per-sample member g (0..7) of the dihedral group: g >= 4 flips first, then rotates by g % 4 turns."""
    out = torch.empty_like(x)
    for k in range(8):
        m = g == k
        if m.any():
            v = x[m].flip(-1) if k >= 4 else x[m]
            out[m] = torch.rot90(v, k % 4, dims=(-2, -1))
    return out


def invert_g(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for k in range(8):
        m = g == k
        if m.any():
            v = torch.rot90(x[m], -(k % 4), dims=(-2, -1))
            out[m] = v.flip(-1) if k >= 4 else v
    return out


def view(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    scale = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.8, 1.25)
    v = apply_g(x, g) * scale
    return v + 0.03 * torch.randn_like(v)


def anchor_loss(f1, f2, proj, per_crop: int, temp: float = 0.1) -> torch.Tensor:
    """InfoNCE over the same base-frame positions in two views; every other anchor (any crop, any place) is a
    negative."""
    b, _, h, w = f1.shape
    ys = torch.arange(MARGIN, h - MARGIN, STRIDE, device=f1.device)
    xs = torch.arange(MARGIN, w - MARGIN, STRIDE, device=f1.device)
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), -1).reshape(-1, 2)
    pick = torch.stack([torch.randperm(len(grid), device=f1.device)[:per_crop] for _ in range(b)])
    pos = grid[pick]  # (B, P, 2)
    bi = torch.arange(b, device=f1.device)[:, None].expand(-1, per_crop)

    def take(f):
        z = proj(f)[bi, :, pos[..., 0], pos[..., 1]]  # (B, P, D)
        return F.normalize(z.reshape(-1, z.shape[-1]), dim=1)

    z1, z2 = take(f1), take(f2)
    logits = z1 @ z2.T / temp
    target = torch.arange(len(z1), device=f1.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def pretrain(dtms, steps=3000, batch=32, crop=128, per_crop=64, lr=2e-3, device="mps", seed=0, log=print):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sample = batch_sampler(dtms, rng, crop, device)
    net = UNet().to(device)
    proj = nn.Sequential(nn.Conv2d(32, 64, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 1)).to(device)
    params = [*net.parameters(), *proj.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for step in range(1, steps + 1):
        x = sample(batch)
        g1, g2 = (torch.randint(0, 8, (batch,), device=device) for _ in range(2))
        f1 = invert_g(net.trunk(view(x, g1)), g1)  # both views back in the crop's own frame
        f2 = invert_g(net.trunk(view(x, g2)), g2)
        loss = anchor_loss(f1, f2, proj, per_crop)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step in (1, steps):
            log(f"step {step:5d}/{steps}  loss {loss.item():.3f}  {time.time() - t0:.0f}s", flush=True)
    return net, steps * batch / (time.time() - t0)
