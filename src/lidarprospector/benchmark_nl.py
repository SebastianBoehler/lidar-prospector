"""A second country and a forest: zero-shot scoring in the Utrechtse Heuvelrug (Netherlands).

Terrain: AHN DTM (CC0) at 1 m. Labels: OpenStreetMap tumulus nodes (ODbL), a hit being a candidate within
15 m. Models are trained on Wessex only, so every number here is zero-shot in a new country and landscape.
"""

import argparse

import numpy as np

from .crossval import tiles
from .detect import detect
from .embed import relief_channels
from .features_torch import pick_device
from .labels_osm import point_labels, tumuli
from .models_bench import BUDGETS, fit_gbm, gbm_predict, hits_at, load_all, predict_dense
from .mounds import UNet
from .peaks import peaks_from_prob
from .sources.ahn import NetherlandsAHN

# 3 km cells (RD New, SW corners) in the Heuvelrug with the most mapped tumuli
CELLS = ((153000, 450000), (153000, 447000), (147000, 459000), (150000, 456000))


def load_nl_tile(src, x, y):
    bbox = (x, y, x + 3000, y + 3000)
    dem = src.fetch(bbox, cell=1.0)
    xy = tumuli(bbox, src.epsg)
    near = point_labels(dem.z.shape, dem.x0, dem.y1, dem.cell, xy)
    return {"bbox": bbox, "dtm": dem.z, "chan": relief_channels(dem.z), "near": near, "n": len(xy)}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="output/unet.pt")
    a = ap.parse_args()
    device = pick_device()
    src = NetherlandsAHN()
    data = [load_nl_tile(src, x, y) for x, y in CELLS]
    for t in data:
        print(f"{t['bbox'][:2]}: {t['n']} mapped tumuli, {np.mean(~np.isnan(t['dtm'])):.0%} terrain")
    targets = sum(t["n"] for t in data)
    model = UNet().to(device)
    model.load_state_dict(torch.load(a.weights, map_location=device))
    clf = fit_gbm(load_all(tiles()), np.random.default_rng(0))
    rng = np.random.default_rng(0)
    rows = {}

    def add(name, hits):
        acc = rows.setdefault(name, dict.fromkeys(BUDGETS, 0.0))
        for k in BUDGETS:
            acc[k] += hits[k]

    for t in data:
        cands = sorted(detect(t["dtm"])[0], key=lambda c: -c.score)
        hand = np.array([(min(int(c.row), 2999), min(int(c.col), 2999)) for c in cands]).reshape(-1, 2)
        add("hand-tuned", hits_at(hand, t))
        add("pixel GBM (Wessex)", hits_at(peaks_from_prob(gbm_predict(clf, t)), t))
        add("U-Net (Wessex)", hits_at(peaks_from_prob(predict_dense(model, t["chan"], device)), t))
        rnd = dict.fromkeys(BUDGETS, 0.0)
        for _ in range(30):
            pts = np.c_[rng.integers(0, 3000, max(BUDGETS)), rng.integers(0, 3000, max(BUDGETS))]
            for k, v in hits_at(pts, t).items():
                rnd[k] += v / 30
        add("random", rnd)
    print(f"\nZERO-SHOT, Netherlands (forest and heath), {targets} OSM tumuli")
    print(f"{'method':<22}" + "".join(f"K={k:<6}" for k in BUDGETS))
    for name, per_k in rows.items():
        print(f"{name:<22}" + "".join(f"{per_k[k] / targets:<8.0%}" for k in BUDGETS))


if __name__ == "__main__":
    main()
