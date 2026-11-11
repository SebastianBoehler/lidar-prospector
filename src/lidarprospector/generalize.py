"""Do the models carry over to landscapes they were not trained on?

Zero-shot: train on the five Wessex tiles only, score each moorland / Peak District tile.
With landscape data: train on Wessex plus the *other* tiles of the new landscapes (leave-one-tile-out).
"""

import argparse

import numpy as np

from .crossval import landscape_tiles, tiles
from .detect import detect
from .features_torch import pick_device
from .models_bench import BUDGETS, fit_gbm, gbm_predict, hits_at, load_all, predict_dense, train_unet
from .peaks import peaks_from_prob


def hand_peaks(test):
    cands = sorted(detect(test["dtm"])[0], key=lambda c: -c.score)
    h, w = test["dtm"].shape
    return np.array([(min(int(c.row), h - 1), min(int(c.col), w - 1)) for c in cands]).reshape(-1, 2)


def random_hits(test, rng, trials=30):
    h, w = test["dtm"].shape
    out = dict.fromkeys(BUDGETS, 0.0)
    for _ in range(trials):
        pts = np.c_[rng.integers(0, h, max(BUDGETS)), rng.integers(0, w, max(BUDGETS))]
        for k, v in hits_at(pts, test).items():
            out[k] += v / trials
    return out


def report(title, rows, targets):
    print(f"\n{title}  ({targets} barrows/cairns)")
    print(f"{'method':<26}" + "".join(f"K={k:<6}" for k in BUDGETS))
    for name, per_k in rows.items():
        print(f"{name:<26}" + "".join(f"{per_k[k] / targets:<8.0%}" for k in BUDGETS))


def add(rows, name, hits):
    acc = rows.setdefault(name, dict.fromkeys(BUDGETS, 0.0))
    for k in BUDGETS:
        acc[k] += hits[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    device = pick_device()
    rng = np.random.default_rng(0)
    wessex = load_all(tiles())
    names, boxes = zip(*landscape_tiles(), strict=True)
    new = load_all(list(boxes))
    for n, t in zip(names, new, strict=True):
        print(f"{n} {t['bbox'][:2]}: {t['n']} barrows/cairns")
    targets = sum(t["n"] for t in new)

    zero: dict = {}
    model = train_unet(wessex, None, a.steps, device, seed=0)
    clf = fit_gbm(wessex, rng)
    for t in new:
        add(zero, "hand-tuned", hits_at(hand_peaks(t), t))
        add(zero, "pixel GBM", hits_at(peaks_from_prob(gbm_predict(clf, t)), t))
        add(zero, "U-Net (Wessex only)", hits_at(peaks_from_prob(predict_dense(model, t["chan"], device)), t))
        add(zero, "random", random_hits(t, rng))
    report("ZERO-SHOT: trained on Wessex only", zero, targets)

    mixed: dict = {}
    for i, t in enumerate(new):
        train = wessex + [o for j, o in enumerate(new) if j != i]
        m = train_unet(train, None, a.steps, device, seed=i)
        h = hits_at(peaks_from_prob(predict_dense(m, t["chan"], device)), t)
        add(mixed, "U-Net (+ other moor/Peak tiles)", h)
        zh = hits_at(peaks_from_prob(predict_dense(model, t["chan"], device)), t)
        line = f"{names[i]} {t['bbox'][:2]}: {t['n']} barrows -> K=200 hits: "
        print(line + f"zero-shot {zh[200]:.0f}, with landscape data {h[200]:.0f}", flush=True)
    mixed["U-Net (Wessex only)"] = zero["U-Net (Wessex only)"]
    mixed["random"] = zero["random"]
    report("WITH LANDSCAPE DATA: leave-one-tile-out over Wessex + new tiles", mixed, targets)


if __name__ == "__main__":
    main()
