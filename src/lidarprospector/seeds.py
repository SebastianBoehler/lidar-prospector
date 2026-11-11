"""Seed noise and the architecture ablation, answered on enough labels to mean something.

Every variant is trained on the five Wessex tiles with several seeds and scored zero-shot on the eight
moorland / Peak District / Dartmoor tiles (135 barrows and cairns). Reports mean and spread per variant.
"""

import argparse
import time

import numpy as np

from .crossval import landscape_tiles, tiles
from .features_torch import pick_device
from .models_bench import BUDGETS, hits_at, load_all, predict_dense, train_unet
from .peaks import peaks_from_prob
from .unet import scaled_widths

VARIANTS = {
    "control: 1500 steps, 1.0x": {"steps": 1500},
    "longer: 4000 steps": {"steps": 4000},
    "wider: 1.5x channels": {"steps": 1500, "widths": scaled_widths(1.5)},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    device = pick_device()
    wessex = load_all(tiles())
    new = load_all([b for _, b in landscape_tiles()])
    targets = sum(t["n"] for t in new)
    recall: dict[str, list[dict[int, float]]] = {v: [] for v in VARIANTS}
    for seed in range(a.seeds):
        for name, cfg in VARIANTS.items():
            t0 = time.time()
            arch = {k: v for k, v in cfg.items() if k == "widths"}
            model = train_unet(wessex, None, cfg["steps"], device, seed=seed, **arch)
            tot = dict.fromkeys(BUDGETS, 0)
            for t in new:
                h = hits_at(peaks_from_prob(predict_dense(model, t["chan"], device)), t)
                for k in BUDGETS:
                    tot[k] += h[k]
            recall[name].append({k: tot[k] / targets for k in BUDGETS})
            print(f"seed {seed} {name}: K=25 {tot[25] / targets:.0%}  K=200 {tot[200] / targets:.0%} "
                  f"({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
    print(f"\nZERO-SHOT on {targets} barrows/cairns, mean +/- sd over {a.seeds} seeds")
    print(f"{'variant':<28}" + "".join(f"K={k:<12}" for k in BUDGETS))
    for name, runs in recall.items():
        cells = []
        for k in BUDGETS:
            v = np.array([r[k] for r in runs])
            cells.append(f"{v.mean():.0%} +/-{v.std():.0%}".ljust(13))
        print(f"{name:<28}" + "".join(cells))


if __name__ == "__main__":
    main()
