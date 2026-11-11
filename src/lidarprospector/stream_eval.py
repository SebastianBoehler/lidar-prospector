"""If only a fraction of the area is fetched at full resolution, how much of the result survives?

A cheap coarse pass scores every block; only the best fraction is fetched fine. Scored zero-shot on the eight
moor / Peak / Dartmoor tiles and the four Dutch tiles (181 labels), with the mound network trained on Wessex.
The fine result is the ranked top-K candidates (K = 22 per km2) found inside the fetched blocks.
"""

import argparse

import numpy as np
import torch
from scipy import ndimage

from .benchmark_nl import CELLS, load_nl_tile
from .crossval import landscape_tiles
from .embed import relief_channels
from .features_torch import pick_device
from .models_bench import load_all, predict_dense
from .peaks import peaks_from_prob
from .sources.ahn import NetherlandsAHN
from .tiling import downsample
from .unet import UNet

BLOCK = 256  # metres (fine pixels) per fetch block
FRACTIONS = (0.1, 0.2, 0.3, 0.5, 1.0)


def block_max(a: np.ndarray, block_px: int) -> np.ndarray:
    h, w = (a.shape[0] // block_px) * block_px, (a.shape[1] // block_px) * block_px
    return np.nan_to_num(a[:h, :w].reshape(h // block_px, block_px, w // block_px, block_px).max(axis=(1, 3)))


def energy_scorer(zs: np.ndarray, block_px: int) -> np.ndarray:
    """No model, no labels: how much relief is there at this scale? Flat fields score zero."""
    f = np.where(np.isnan(zs), np.nanmean(zs), zs)
    return block_max(np.abs(f - ndimage.gaussian_filter(f, 12.0)), block_px)


def model_scorer(model, device):
    def score(zs, block_px):
        return block_max(predict_dense(model, relief_channels(zs), device), block_px)

    return score


def evaluate(tile, prob, score_fn, scale, rng):
    """Recall of the tile's labels in the top-K candidates, for each fetched fraction (+ random blocks)."""
    h, w = prob.shape
    pk = peaks_from_prob(prob)
    zs = downsample(tile["dtm"], scale)
    scores = score_fn(zs, BLOCK // scale)
    nb_r, nb_c = scores.shape
    k = round(22 * h * w / 1e6)
    br = np.minimum(pk[:, 0] // BLOCK, nb_r - 1)
    bc = np.minimum(pk[:, 1] // BLOCK, nb_c - 1)
    out = {}
    for f in FRACTIONS:
        n = max(1, round(f * scores.size))
        for kind in ("scored", "random"):
            order = np.argsort(-scores.ravel()) if kind == "scored" else rng.permutation(scores.size)
            chosen = np.zeros(scores.size, bool)
            chosen[order[:n]] = True
            keep = pk[chosen[br * nb_c + bc]][:k]
            found = {int(tile["near"][r, c]) for r, c in keep} - {0}
            out[(kind, f)] = len(found)
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    device = pick_device()
    model = UNet().to(device)
    model.load_state_dict(torch.load("output/unet.pt", map_location=device))
    tiles_ = load_all([b for _, b in landscape_tiles()])
    src = NetherlandsAHN()
    tiles_ += [load_nl_tile(src, x, y) for x, y in CELLS]
    targets = sum(t["n"] for t in tiles_)
    probs = [predict_dense(model, t["chan"], device) for t in tiles_]
    rng = np.random.default_rng(0)
    scorers = {
        "relief energy @4 m (no model)": (energy_scorer, 4),
        "mound net @2 m": (model_scorer(model, device), 2),
        "mound net @4 m": (model_scorer(model, device), 4),
    }
    print(f"{targets} labelled barrows/cairns/tumuli in {len(tiles_)} tiles; blocks of {BLOCK} m")
    print(
        f"{'coarse scorer (fraction of blocks fetched at full res)':<46}"
        + "".join(f"{f:>7.0%}" for f in FRACTIONS)
    )
    base = None
    for name, (fn, s) in scorers.items():
        acc = {}
        for t, p in zip(tiles_, probs, strict=True):
            for key, v in evaluate(t, p, fn, s, rng).items():
                acc[key] = acc.get(key, 0) + v
        full = acc[("scored", 1.0)]
        base = base or acc
        print(f"{name:<46}" + "".join(f"{acc[('scored', f)] / full:>7.0%}" for f in FRACTIONS))
    print(
        f"{'random blocks (control)':<46}"
        + "".join(f"{base[('random', f)] / base[('scored', 1.0)]:>7.0%}" for f in FRACTIONS)
    )
    print(f"(100% = {base[('scored', 1.0)]}/{targets} labels found fetching everything at full resolution)")


if __name__ == "__main__":
    main()
