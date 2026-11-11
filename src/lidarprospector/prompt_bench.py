"""Simulated-user benchmark for prompt-by-example search.

The 'user' marks N known barrows as examples; the system ranks every other place in the tile; recall is
the share of the tile's remaining scheduled barrows found within the top K. With examples from the same
tile, the user can then accept or reject the top results (oracle: a hit within 10 m of a scheduled
barrow) and the system re-ranks.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .crossval import tiles
from .embed import Encoder, embed_grid, relief_channels
from .labels import monuments
from .learn import rasters
from .retrieve import Index
from .wcs import fetch_dtm

BUDGETS = (25, 50, 100, 200)


def load_tile(bbox, model, device):
    dtm, x0, y1 = fetch_dtm(*bbox)
    tight, ids, _, _, near = rasters(dtm, x0, y1, monuments(*bbox))
    n = len(tight)
    centres = np.array(ndimage.center_of_mass(ids > 0, ids, range(1, n + 1))) if n else np.zeros((0, 2))
    index = Index(embed_grid(model, relief_channels(dtm), device=device))
    return {"bbox": bbox, "index": index, "near": near, "centres": centres, "n": n}


def example_vec(tile, i):
    idx = tile["index"]
    return idx.flat[idx.cell_of(*tile["centres"][i])]


def unit(v):
    return v / np.linalg.norm(v)


def trial(tile, others, n_ex, source, rng):
    """One simulated session. Returns {method: {K: (hits, n_targets)}}."""
    idx, near, n = tile["index"], tile["near"], tile["n"]
    if source == "same":
        pick = rng.choice(n, n_ex, replace=False)
        ex_cells = [idx.cell_of(*tile["centres"][i]) for i in pick]
        ex_vecs = idx.flat[ex_cells]
        targets = set(range(1, n + 1)) - {int(i) + 1 for i in pick}
        exclude = set(ex_cells)
    else:
        pool = [(o, i) for o in others for i in range(o["n"])]
        ex_vecs = np.stack([example_vec(*pool[p]) for p in rng.choice(len(pool), n_ex, replace=False)])
        ex_cells, targets, exclude = [], set(range(1, n + 1)), set()

    out = {}

    def found(cells):
        return {int(near[idx.pixel_of(c)]) for c in cells} - {0}

    def evaluate(name, scores):
        top = idx.top(scores, max(BUDGETS), exclude=exclude)
        out[name] = {k: (len(found(top[:k]) & targets), len(targets)) for k in BUDGETS}

    scores = idx.flat @ unit(ex_vecs.mean(0))
    evaluate("prototype", scores)
    evaluate("random", rng.random(len(idx.flat)))
    if source == "same":
        pos, neg = list(ex_cells), []
        for rnd in (1, 2):
            shown = idx.top(scores, 20, exclude=exclude | set(pos) | set(neg))
            good = [c for c in shown if found([c]) & targets]
            pos += good
            neg += [c for c in shown if c not in good]
            scores = idx.classifier_scores(pos, neg, rng)
            evaluate(f"feedback x{rnd}", scores)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=Path, default=Path("output/encoder.pt"))
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Encoder().to(device)
    model.load_state_dict(torch.load(a.encoder, map_location=device))
    data = [load_tile(b, model, device) for b in tiles()]
    big = [t for t in data if t["n"] >= 10]
    print(f"tiles scored (>=10 barrows): {[(t['bbox'][0], t['bbox'][1], t['n']) for t in big]}")
    rng = np.random.default_rng(0)
    for source in ("same", "other"):
        label = "the same tile (held out of scoring)" if source == "same" else "other tiles"
        print(f"\nexamples from {label}")
        for n_ex in (1, 3, 5):
            agg = {}
            for t in big:
                others = [o for o in data if o is not t and o["n"] > 0]
                for _ in range(a.trials):
                    for m, per_k in trial(t, others, n_ex, source, rng).items():
                        for k, (h, n) in per_k.items():
                            ah, an = agg.setdefault(m, {}).get(k, (0, 0))
                            agg[m][k] = (ah + h, an + n)
            print(f"  {n_ex} example(s):   " + "  ".join(f"K={k:<4}" for k in BUDGETS))
            for m, per_k in agg.items():
                print(f"    {m:<12} " + "  ".join(f"{per_k[k][0] / per_k[k][1]:6.0%}" for k in BUDGETS))


if __name__ == "__main__":
    main()
