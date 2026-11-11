"""Why does the U-Net work? Vary one thing at a time (training length, width, input features) on the
three tiles with enough barrows, leave-one-tile-out, same metric as `models_bench`."""

import argparse
import time

import numpy as np
import torch

from .features_torch import feature_stack_torch, pick_device
from .models_bench import BUDGETS, hits_at, load_all, predict_dense, train_unet
from .peaks import peaks_from_prob
from .unet import scaled_widths


def add_feature_channels(data):
    """relief (2) + the 13 terrain features, each divided by its typical spread across tiles."""
    feats = [feature_stack_torch(t["dtm"])[0] for t in data]
    spread = np.median([np.nanstd(f.reshape(-1, f.shape[-1]), axis=0) for f in feats], axis=0)
    for t, f in zip(data, feats, strict=True):
        f = np.clip(np.nan_to_num(f / np.maximum(spread, 1e-6)), -6, 6)
        t["chan15"] = np.concatenate([t["chan"], np.moveaxis(f, -1, 0)]).astype("float32")


VARIANTS = {
    "control: 1500 steps, 2 ch, 1.0x": {"steps": 1500},
    "longer: 4000 steps": {"steps": 4000},
    "wider: 1.5x channels": {"steps": 1500, "widths": scaled_widths(1.5)},
    "inputs: relief + 13 terrain features": {"steps": 1500, "key": "chan15", "in_ch": 15},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    device = pick_device()
    data = load_all()
    add_feature_channels(data)
    totals = {name: dict.fromkeys(BUDGETS, 0) for name in VARIANTS}
    targets = 0
    for i in a.folds:
        test, train = data[i], [t for j, t in enumerate(data) if j != i]
        targets += test["n"]
        for name, cfg in VARIANTS.items():
            t0 = time.time()
            key = cfg.get("key", "chan")
            arch = {k: v for k, v in cfg.items() if k in ("in_ch", "widths")}
            model = train_unet(train, None, cfg["steps"], device, seed=i, key=key, **arch)
            prob = predict_dense(model, test[key], device)
            h = hits_at(peaks_from_prob(prob), test)
            for k in BUDGETS:
                totals[name][k] += h[k]
            print(f"fold {i} {name}: K=200 {h[200]}/{test['n']} ({time.time() - t0:.0f}s)", flush=True)
            del model
            torch.mps.empty_cache()
    print(f"\nPOOLED over {targets} held-out barrows (K candidates per tile)")
    print(f"{'variant':<40}" + "".join(f"K={k:<6}" for k in BUDGETS))
    for name, per_k in totals.items():
        print(f"{name:<40}" + "".join(f"{per_k[k] / targets:<8.0%}" for k in BUDGETS))


if __name__ == "__main__":
    main()
