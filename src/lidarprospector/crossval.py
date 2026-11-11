"""Leave-one-tile-out: train the barrow classifier on four tiles, score it on the fifth."""

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from sklearn.ensemble import HistGradientBoostingClassifier

from .detect import detect
from .features import feature_stack
from .labels import monuments
from .learn import BUDGETS, rasters, recall_at
from .wcs import fetch_dtm

STONEHENGE = (410800, 140800, 413800, 143800)


# 3 km cells chosen by label density from the Wessex chalk scan (two barrow-dense, two with
# linear/enclosure/field monuments), SW corners in EPSG:27700.
EXTRA_TILES = ((420000, 144000), (417000, 144000), (387000, 117000), (414000, 153000))


def tiles():
    return [STONEHENGE] + [(x, y, x + 3000, y + 3000) for x, y in EXTRA_TILES]


def load(bbox, rng, n_neg=120_000):
    dtm, x0, y1 = fetch_dtm(*bbox)
    tight, ids, pos, ignore, near = rasters(dtm, x0, y1, monuments(*bbox))
    F, _ = feature_stack(dtm)
    Ff = F.reshape(-1, F.shape[-1])
    valid = ~np.isnan(dtm)
    p = np.flatnonzero(pos & valid)
    n = rng.choice(np.flatnonzero(~pos & ~ignore & valid), n_neg, replace=False)
    return {
        "bbox": bbox,
        "dtm": dtm,
        "ids": ids,
        "near": near,
        "n_barrows": len(tight),
        "X": np.concatenate([Ff[p], Ff[n]]),
        "y": np.r_[np.ones(len(p)), np.zeros(len(n))],
    }


def main() -> None:
    rng = np.random.default_rng(0)
    data = [load(b, rng) for b in tiles()]
    print("tile (E,N)         barrows  train px (pos)")
    for d in data:
        print(f"{d['bbox'][0]},{d['bbox'][1]:<9} {d['n_barrows']:7d}  {len(d['y']):7d} ({int(d['y'].sum())})")
    pooled = {k: [0, 0, 0, 0] for k in BUDGETS}  # learned, hand, random hits ; total
    total = 0
    for i, te in enumerate(data):
        tr = [d for j, d in enumerate(data) if j != i]
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.08, max_leaf_nodes=31, random_state=0
        ).fit(np.concatenate([d["X"] for d in tr]), np.concatenate([d["y"] for d in tr]))
        dtm, near = te["dtm"], te["near"]
        F, _ = feature_stack(dtm)
        prob = clf.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(dtm.shape).astype("float32")
        prob = ndimage.gaussian_filter(prob, 1.5)
        pk = peak_local_max(prob, min_distance=6, threshold_abs=0.02, exclude_border=False)
        pk = pk[np.argsort(-prob[pk[:, 0], pk[:, 1]])]
        hand = sorted(detect(dtm)[0], key=lambda c: -c.score)
        h_rc = [(min(int(c.row), dtm.shape[0] - 1), min(int(c.col), dtm.shape[1] - 1)) for c in hand]
        ids = {int(v) for v in np.unique(te["ids"]) if v}
        total += len(ids)
        for k in BUDGETS:
            rnd = np.mean(
                [
                    len(
                        {
                            near[r, c]
                            for r, c in np.c_[
                                rng.integers(0, dtm.shape[0], k), rng.integers(0, dtm.shape[1], k)
                            ]
                        }
                        - {0}
                        & ids
                    )
                    for _ in range(30)
                ]
            )
            pooled[k][0] += recall_at(pk, near, ids, k) * len(ids)
            pooled[k][1] += recall_at(h_rc, near, ids, k) * len(ids)
            pooled[k][2] += rnd
        print(
            f"held out {te['bbox'][:2]}: {len(ids)} barrows, "
            f"learned recall@200 {recall_at(pk, near, ids, 200):.0%}, "
            f"hand {recall_at(h_rc, near, ids, 200):.0%}",
            flush=True,
        )
    print(f"\nPOOLED over {total} held-out barrows (K candidates per tile)")
    print("  K     learned   hand-tuned   random")
    for k in BUDGETS:
        l, h, r, _ = pooled[k]
        print(f"  {k:<4} {l / total:8.0%} {h / total:11.0%} {r / total:8.0%}")


if __name__ == "__main__":
    main()
