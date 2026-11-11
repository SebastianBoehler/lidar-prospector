"""The 13-channel feature stack as separable convolutions in PyTorch, so it runs on whatever
accelerator is present (Metal, CUDA) or the CPU. Same channels as `features.feature_stack`; the one
intended difference is reflect padding at tile edges instead of skimage's zero padding for the Hessian,
which makes edge pixels better-behaved. Interior pixels agree with the NumPy version."""

import numpy as np
import torch
import torch.nn.functional as F

from .features import HESSIAN_SIGMAS, RELIEF_SIGMAS, fill_nodata


def _gauss(sigma: float, order: int, device) -> torch.Tensor:
    """Gaussian or its 1st/2nd derivative as a discrete kernel. Derivative kernels are re-normalised
    so a constant has exactly zero response and a ramp / parabola has unit slope / curvature; without
    that, truncation leaks the mean elevation (100+ m) into the curvature channels."""
    r = int(4.0 * sigma + 0.5)
    x = torch.arange(-r, r + 1, dtype=torch.float64, device="cpu")
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    if order == 1:
        k = x / sigma**2 * g
        k = k / (k * x).sum()
    elif order == 2:
        k = (x**2 / sigma**4 - 1.0 / sigma**2) * g
        k = k - k.mean()
        k = 2.0 * k / (k * x**2).sum()
    else:
        k = g
    return k.to(device=device, dtype=torch.float32)


def _sep(img: torch.Tensor, ky: torch.Tensor, kx: torch.Tensor) -> torch.Tensor:
    """Separable 2-D filter of (1, 1, H, W) with reflect padding."""
    ry, rx = ky.numel() // 2, kx.numel() // 2
    if ry >= img.shape[2] or rx >= img.shape[3]:
        raise ValueError(
            f"tile {tuple(img.shape[2:])} is smaller than the filter radius ({ry} px); use a larger tile"
        )
    out = F.pad(img, (0, 0, ry, ry), mode="reflect")
    out = F.conv2d(out, ky.view(1, 1, -1, 1))
    out = F.pad(out, (rx, rx, 0, 0), mode="reflect")
    return F.conv2d(out, kx.view(1, 1, 1, -1))


def _smooth(img, sigma, device):
    g = _gauss(sigma, 0, device)
    return _sep(img, g, g)


def _box(img, size, device):
    k = torch.full((size,), 1.0 / size, device=device)
    return _sep(img, k, k)


def pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def feature_stack_torch(dtm: np.ndarray, device: str = "auto"):
    """(H, W, F) float32 plus names; NaN where the input is NaN."""
    device = pick_device(device)
    filled = fill_nodata(dtm)
    # every channel is offset-invariant, so remove the mean to keep float32 cancellation error small
    z = torch.from_numpy((filled - filled.mean()).astype("float32")).to(device)[None, None]
    chans, names = [], []
    for s in RELIEF_SIGMAS:
        chans.append(_smooth(z, s, device) - _smooth(z, 3 * s, device))
        names.append(f"relief{s}")
    for s in HESSIAN_SIGMAS:
        g0, g2 = _gauss(s, 0, device), _gauss(s, 2, device)
        hrr, hcc = _sep(z, g2, g0), _sep(z, g0, g2)
        hrc = _sep(z, _gauss(s, 1, device), _gauss(s, 1, device))
        mean = 0.5 * (hrr + hcc)
        diff = torch.sqrt((0.5 * (hrr - hcc)) ** 2 + hrc**2)
        chans += [(mean + diff) * s**2, (mean - diff) * s**2]  # l1 >= l2
        names += [f"eig1_{s}", f"eig2_{s}"]
    sm = _smooth(z, 2.0, device)
    gy = torch.gradient(sm[0, 0], dim=0)[0][None, None]
    gx = torch.gradient(sm[0, 0], dim=1)[0][None, None]
    slope = torch.hypot(gy, gx)
    chans.append(slope)
    names.append("slope2")
    chans.append(torch.sqrt(torch.clamp(_box(slope**2, 21, device), min=0.0)))
    names.append("rough10")
    chans.append(_smooth(z, 25.0, device) - _smooth(z, 75.0, device))
    names.append("regional")
    out = torch.cat(chans, dim=1)[0].permute(1, 2, 0).cpu().numpy()
    out[np.isnan(dtm)] = np.nan
    return out, names
