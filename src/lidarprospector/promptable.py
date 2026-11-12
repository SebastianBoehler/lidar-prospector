"""Prompt-driven detection with no per-type training.

A prompt is a handful of example locations ("this is what I mean"). Every pixel gets a dense feature vector
from a general-purpose extractor; a linear probe separates the prompt's examples from background pixels; the
probe scores any terrain, and its peaks are the candidates. Swapping the examples swaps the object type.

Extractors (all dense, one vector per 1 m pixel):
  hand  : the 13 multi-scale terrain descriptors (relief, curvature, slope, roughness)
  unet  : the decoder features of the barrow-trained U-Net (what a supervised model learned to see)
  ssl   : the self-supervised encoder's multi-scale feature maps, projected to 32 dims (no labels used)
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from .embed import Encoder, relief_channels
from .features_torch import feature_stack_torch, pick_device
from .unet import UNet


@dataclass
class Extractor:
    name: str
    dense: Callable[[np.ndarray], np.ndarray]  # dtm (H, W) -> features (H, W, D) float32
    dim: int


def blockwise(fn, chan: np.ndarray, d_out: int, device: str, block: int = 1024, pad: int = 32) -> np.ndarray:
    """Apply a fully convolutional `fn` (B, C, h, w) -> (B, d_out, h, w) over a big image in blocks."""
    _, h, w = chan.shape
    ph, pw = (-h) % block, (-w) % block
    x = F.pad(torch.from_numpy(chan)[None], (pad, pad, pad, pad), mode="reflect")
    x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    out = np.zeros((h + ph, w + pw, d_out), np.float32)
    with torch.no_grad():
        for r in range(0, h + ph, block):
            for c in range(0, w + pw, block):
                tile = x[:, :, r : r + block + 2 * pad, c : c + block + 2 * pad].to(device)
                feats = fn(tile)[0, :, pad:-pad, pad:-pad]
                out[r : r + block, c : c + block] = feats.permute(1, 2, 0).cpu().numpy()
    return out[:h, :w]


def hand_extractor() -> Extractor:
    return Extractor("hand terrain features", lambda dtm: feature_stack_torch(dtm)[0], 13)


def unet_extractor(weights: str = "output/unet.pt") -> Extractor:
    device = pick_device()
    model = UNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return Extractor(
        "U-Net decoder features",
        lambda dtm: blockwise(model.trunk, relief_channels(dtm), 32, device),
        32,
    )


def pca_extractor(name: str, raw_fn, raw_dim: int, device: str, dim: int = 32) -> Extractor:
    """Dense features from `raw_fn` (B, C, h, w) -> (B, raw_dim, h, w), projected to `dim` by a PCA fitted
    once on a crop of the first terrain seen, applied on the device inside each block."""
    state: dict = {}

    def projected(x):
        return torch.einsum("bchw,cd->bdhw", raw_fn(x) - state["mean"], state["proj"])

    def dense(dtm):
        chan = relief_channels(dtm)
        if "proj" not in state:
            crop = np.ascontiguousarray(chan[:, :1024, :1024])
            raw = blockwise(raw_fn, crop, raw_dim, device)
            sample = raw.reshape(-1, raw_dim)[:: max(1, raw.shape[0] * raw.shape[1] // 40000)]
            mean = sample.mean(0)
            _, _, vt = np.linalg.svd(sample - mean, full_matrices=False)
            state["mean"] = torch.from_numpy(mean).to(device)[None, :, None, None]
            state["proj"] = torch.from_numpy(vt[:dim].T.copy()).to(device)
        return blockwise(projected, chan, dim, device)

    return Extractor(name, dense, dim)


def multiscale_taps(body, taps=(0, 2, 4), extra_zero_channels: int = 0):
    """raw_fn for `pca_extractor`: the body's feature maps at several depths, upsampled and concatenated."""

    def fn(x):
        if extra_zero_channels:
            x = torch.cat([x, torch.zeros_like(x[:, :1]).repeat(1, extra_zero_channels, 1, 1)], 1)
        feats = []
        for i, block in enumerate(body):
            x = block(x)
            if i in taps:
                feats.append(x)
        full = feats[0].shape[-2:]
        return torch.cat(
            [F.interpolate(f, size=full, mode="bilinear", align_corners=False) for f in feats], 1
        )

    return fn


def ssl_extractor(weights: str = "output/encoder.pt", dim: int = 32) -> Extractor:
    device = pick_device()
    enc = Encoder().to(device)
    enc.load_state_dict(torch.load(weights, map_location=device))
    enc.eval()
    return pca_extractor("SSL encoder features (PCA 32)", multiscale_taps(enc.body), 224, device, dim)


def fit_probe(pos: np.ndarray, neg: np.ndarray, c: float = 0.05) -> tuple[np.ndarray, float]:
    """Linear separator of the prompt's feature vectors from background, strongly regularised: a prompt is a
    handful of examples, so the model has to generalise from very little."""
    x = np.concatenate([pos, neg])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=300).fit(x, y)
    return clf.coef_[0].astype(np.float32), float(clf.intercept_[0])


def score_maps(
    feats: torch.Tensor, weights: np.ndarray, bias: np.ndarray, shape, pool: int = 2
) -> torch.Tensor:
    """(M, h, w) probe scores for M prompts from standardised features (N, D); average-pooled by `pool`."""
    w = torch.from_numpy(weights).to(feats.device)
    b = torch.from_numpy(bias).to(feats.device)
    h, wd = shape
    out = torch.empty(len(weights), h, wd, device=feats.device)
    step = 1 << 20
    flat = out.view(len(weights), -1)
    for i in range(0, feats.shape[0], step):
        flat[:, i : i + step] = (feats[i : i + step] @ w.T + b).T
    return F.avg_pool2d(out[None], pool)[0] if pool > 1 else out


def relief_band(dtm: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Local relief (terrain minus its smooth trend): what a template should match, not absolute height."""
    from scipy import ndimage

    from .features import fill_nodata

    z = fill_nodata(dtm).astype("float64")
    return (z - ndimage.gaussian_filter(z, sigma)).astype(np.float32)


def template_scores(relief: np.ndarray, patches: np.ndarray, device: str, pool: int = 2) -> torch.Tensor:
    """Rotation-aware template matching on raw terrain: for each example patch, the normalised
    cross-correlation with every place (best of the 8 flips/quarter-turns), then the mean over examples.
    No learning, so it finds whatever *shape* the examples share, including shapes no model was trained on."""
    n = patches.shape[-1]
    x = torch.from_numpy(relief)[None, None].to(device)
    ones = torch.ones(1, 1, n, n, device=device)
    s1 = F.conv2d(x, ones, padding=n // 2)
    s2 = F.conv2d(x * x, ones, padding=n // 2)
    energy = torch.sqrt(torch.clamp(s2 - s1 * s1 / (n * n), min=1e-6))
    total = None
    for patch in patches:
        t = torch.from_numpy(patch - patch.mean()).to(device)
        variants = [torch.rot90(v, k) for v in (t, t.flip(-1)) for k in range(4)]
        ker = torch.stack([v / (v.norm() + 1e-6) for v in variants])[:, None]
        ncc = F.conv2d(x, ker, padding=n // 2) / energy
        best = ncc.max(dim=1).values
        total = best if total is None else total + best
    out = total[0] / len(patches)  # (H, W)
    return F.avg_pool2d(out[None, None], pool)[0, 0] if pool > 1 else out


class PromptSearch:
    """Interactive prompting on one terrain window: dense features are computed once, then every query (a few
    clicked examples, optionally some rejected places) fits a probe and returns ranked candidates."""

    def __init__(self, extractor: Extractor, dtm: np.ndarray, device: str, seed: int = 0, n_bg: int = 20000):
        feats = extractor.dense(dtm)
        self.shape = dtm.shape
        d = feats.shape[-1]
        rng = np.random.default_rng(seed)
        self.rng = rng
        h, w = self.shape
        bg = feats[rng.integers(0, h, n_bg), rng.integers(0, w, n_bg)]
        self.mean, self.std = bg.mean(0), bg.std(0) + 1e-6
        self.bg = (bg - self.mean) / self.std
        self.feats = torch.from_numpy(((feats - self.mean) / self.std).reshape(-1, d)).to(device)
        self.valid = ~np.isnan(dtm)

    def vectors(self, points) -> np.ndarray:
        """Standardised features at (row, col) points and their 3x3 neighbourhoods: (n * 9, D)."""
        h, w = self.shape
        out = []
        for r, c in points:
            r, c = int(np.clip(round(r), 1, h - 2)), int(np.clip(round(c), 1, w - 2))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    out.append(self.feats[(r + dr) * w + c + dc].cpu().numpy())
        return np.array(out)

    def search(self, pos, neg=(), k: int = 30, pool: int = 2) -> list[tuple[int, int]]:
        """Top-k (row, col) candidates: a probe fitted on `pos` against `neg` plus random background."""
        from scipy import ndimage
        from skimage.feature import peak_local_max

        negs = np.concatenate([self.bg, self.vectors(neg)]) if len(neg) else self.bg
        wts, b = fit_probe(self.vectors(pos), negs)
        s = score_maps(self.feats, wts[None], np.array([b], np.float32), self.shape, pool)[0].cpu().numpy()
        m = ndimage.gaussian_filter(s, 1.0)
        valid = self.valid[: m.shape[0] * pool : pool, : m.shape[1] * pool : pool][: m.shape[0], : m.shape[1]]
        m = np.where(valid, m, m.min())
        pk = peak_local_max(m, min_distance=3, num_peaks=k + 3 * len(pos), exclude_border=False)
        cand = [(int(r) * pool + pool // 2, int(c) * pool + pool // 2) for r, c in pk]
        taken = np.array([(r, c) for r, c in [*pos, *neg]]) if len(pos) + len(neg) else np.zeros((0, 2))
        keep = [p for p in cand if not len(taken) or np.hypot(*(taken - p).T).min() > 8]
        return keep[:k]
