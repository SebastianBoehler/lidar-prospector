"""Self-supervised terrain encoder (SimCLR-style). Needs the optional `embed` extra (PyTorch).

No labels are used: two augmented views of the same terrain patch must embed close together and
away from other patches. The augmentations (flips, 90-degree rotations, shifts, amplitude jitter)
teach rotation- and shift-invariant shape descriptors, so "what does this patch look like" becomes a
vector that can be searched by example.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn

from .features import fill_nodata

PATCH = 64  # metres at 1 m cells
DIM = 128


def relief_channels(dtm):
    """(2, H, W) float32: broad relief (mounds, banks) and fine relief (ring ditches), scaled to ~[-1, 1]."""
    z = fill_nodata(dtm).astype("float64")
    broad = np.clip((ndimage.gaussian_filter(z, 1.0) - ndimage.gaussian_filter(z, 12.0)) / 2.0, -1, 1)
    fine = np.clip((ndimage.gaussian_filter(z, 0.7) - ndimage.gaussian_filter(z, 4.0)) / 0.5, -1, 1)
    return np.stack([broad, fine]).astype("float32")


def _block(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            _block(2, 32, 1),
            _block(32, 64, 2),
            _block(64, 64, 1),
            _block(64, 128, 2),
            _block(128, 128, 1),
            _block(128, 192, 2),
        )
        self.out = nn.Linear(2 * 192, DIM)
        self.proj = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, 64))

    def forward(self, x):
        h = self.body(x)
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        return F.normalize(self.out(h), dim=1)


def gather(tiles, t, y, x, size=PATCH):
    """Batched crops tiles[t, :, y:y+size, x:x+size] for per-sample t, y, x."""
    r = torch.arange(size, device=tiles.device)
    c = torch.arange(tiles.shape[1], device=tiles.device)
    return tiles[
        t[:, None, None, None],
        c[None, :, None, None],
        (y[:, None, None, None] + r[None, None, :, None]),
        (x[:, None, None, None] + r[None, None, None, :]),
    ]


def dihedral(x):
    """Random member of the 8-element rotation/flip group, applied per sample."""
    k = torch.randint(0, 8, (x.shape[0],), device=x.device)
    out = x.clone()
    for g in range(8):
        m = k == g
        if m.any():
            v = x[m].flip(-1) if g >= 4 else x[m]
            out[m] = torch.rot90(v, g % 4, dims=(-2, -1))
    return out


def augment(x):
    x = dihedral(x)
    scale = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.8, 1.25)
    return x * scale + 0.03 * torch.randn_like(x)


def nt_xent(z1, z2, temp=0.2):
    z = torch.cat([z1, z2])
    sim = z @ z.T / temp
    sim.fill_diagonal_(float("-inf"))
    n = z1.shape[0]
    target = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(sim, target)


def _sampler(chans, rng, stride=8):
    """Patch origins per tile: half uniform, half weighted by local relief so structured terrain is seen
    often (flat fields would otherwise dominate)."""
    coords, weights = [], []
    for t, ch in enumerate(chans):
        h, w = ch.shape[1:]
        ys, xs = np.mgrid[0 : h - PATCH - 8 : stride, 0 : w - PATCH - 8 : stride]
        c = ch[0]
        var = ndimage.uniform_filter(c**2, PATCH) - ndimage.uniform_filter(c, PATCH) ** 2
        wt = np.sqrt(np.maximum(var[ys + PATCH // 2, xs + PATCH // 2], 0)).ravel() ** 2
        coords.append(np.c_[np.full(ys.size, t), ys.ravel() + 4, xs.ravel() + 4])
        weights.append(wt)
    coords = np.concatenate(coords)
    weights = np.concatenate(weights)
    weights = weights / weights.sum()

    def draw(n):
        half = n // 2
        a = coords[rng.integers(0, len(coords), half)]
        b = coords[rng.choice(len(coords), n - half, p=weights)]
        return np.concatenate([a, b])

    return draw


def train(dtms, steps=3000, batch=256, lr=2e-3, device="mps", seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    chans = [relief_channels(d) for d in dtms]
    tiles = torch.from_numpy(np.stack(chans)).to(device)
    draw = _sampler(chans, rng)
    model = Encoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for step in range(1, steps + 1):
        base = torch.from_numpy(draw(batch)).to(device)
        views = []
        for _ in range(2):
            jy, jx = (torch.randint(-4, 5, (batch,), device=device) for _ in range(2))
            views.append(augment(gather(tiles, base[:, 0], base[:, 1] + jy, base[:, 2] + jx)))
        z1 = model.proj(model(views[0]))
        z2 = model.proj(model(views[1]))
        loss = nt_xent(F.normalize(z1, dim=1), F.normalize(z2, dim=1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == 1:
            print(f"step {step:5d}/{steps}  loss {loss.item():.3f}  {time.time() - t0:.0f}s", flush=True)
    return model


@torch.no_grad()
def embed_grid(model, chan, stride=8, device="mps", chunk=4096):
    """(gh, gw, DIM) embeddings of PATCH-sized windows on a regular grid; cell (i, j) is centred at pixel
    (PATCH/2 + i*stride, PATCH/2 + j*stride)."""
    model.eval()
    x = torch.from_numpy(chan).to(device)
    win = x.unfold(1, PATCH, stride).unfold(2, PATCH, stride)  # (C, gh, gw, P, P)
    c, gh, gw = win.shape[:3]
    flat = win.permute(1, 2, 0, 3, 4).reshape(gh * gw, c, PATCH, PATCH)
    out = torch.cat([model(flat[i : i + chunk]) for i in range(0, flat.shape[0], chunk)])
    return out.reshape(gh, gw, DIM).cpu().numpy()
