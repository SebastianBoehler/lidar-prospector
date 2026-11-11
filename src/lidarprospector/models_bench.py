"""Compare model families on one task (barrow recall at K candidates per tile) with identical
leave-one-tile-out splits: pixel GBM, SSL linear probe, U-Net from scratch, U-Net from the SSL encoder."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .crossval import tiles
from .embed import Encoder, dihedral, embed_grid, gather, relief_channels
from .features_torch import feature_stack_torch
from .labels import monuments
from .learn import rasters
from .mounds import predict_dense
from .peaks import peaks_from_prob
from .retrieve import Index
from .unet import UNet
from .wcs import fetch_dtm

BUDGETS = (25, 50, 100, 200)


def load_all():
    data = []
    for bbox in tiles():
        dtm, x0, y1 = fetch_dtm(*bbox)
        tight, ids, pos, ignore, near = rasters(dtm, x0, y1, monuments(*bbox))
        data.append({"bbox": bbox, "dtm": dtm, "chan": relief_channels(dtm), "pos": pos, "ignore": ignore,
                     "ids": ids, "near": near, "n": len(tight)})  # fmt: skip
    return data


def hits_at(peaks, tile):
    out = {}
    for k in BUDGETS:
        found = {int(tile["near"][r, c]) for r, c in peaks[:k]} - {0}
        out[k] = len(found)
    return out


def gbm_prob(train, test, rng):
    xs, ys = [], []
    for t in train:
        feats, _ = feature_stack_torch(t["dtm"])
        flat = feats.reshape(-1, feats.shape[-1])
        valid = ~np.isnan(t["dtm"])
        p = np.flatnonzero(t["pos"] & valid)
        n = rng.choice(np.flatnonzero(~t["pos"] & ~t["ignore"] & valid), 120_000, replace=False)
        xs += [flat[p], flat[n]]
        ys += [np.ones(len(p)), np.zeros(len(n))]
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, random_state=0)
    clf.fit(np.concatenate(xs), np.concatenate(ys))
    feats, _ = feature_stack_torch(test["dtm"])
    return clf.predict_proba(feats.reshape(-1, feats.shape[-1]))[:, 1].reshape(test["dtm"].shape)


def probe_peaks(train, test, enc, device, rng):
    pos, bg = [], []
    for t in train:
        idx = Index(embed_grid(enc, t["chan"], device=device))
        n = t["n"]
        cen = ndimage.center_of_mass(t["ids"] > 0, t["ids"], range(1, n + 1))
        pos += [idx.flat[idx.cell_of(*c)] for c in cen]
        bg.append(idx.flat[rng.integers(0, len(idx.flat), 3000)])
    x = np.concatenate([np.stack(pos), np.concatenate(bg)])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(x, np.r_[np.ones(len(pos)), np.zeros(len(x) - len(pos))])
    idx = Index(embed_grid(enc, test["chan"], device=device))
    top = idx.top(idx.flat @ clf.coef_[0], max(BUDGETS))
    return np.array([idx.pixel_of(c) for c in top])


def train_unet(train, ssl_state, steps, device, seed, batch=24, crop=128, key="chan", **arch):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    stack = []
    for t in train:
        masks = np.stack([t["pos"], t["ignore"]]).astype("float32")
        stack.append(np.concatenate([t[key], masks]).astype("float32"))
    tiles_t = torch.from_numpy(np.stack(stack)).to(device)
    coords = [np.argwhere(t["pos"]) for t in train]
    h, w = train[0]["dtm"].shape
    model = UNet(ssl_state, **arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    pos_w = torch.tensor(10.0, device=device)
    n_in = train[0][key].shape[0]
    for _ in range(steps):
        ti = rng.integers(0, len(train), batch)
        ys, xs = np.empty(batch, int), np.empty(batch, int)
        for b, t in enumerate(ti):
            if b % 2 == 0 and len(coords[t]):
                cy, cx = coords[t][rng.integers(0, len(coords[t]))] + rng.integers(-48, 49, 2)
            else:
                cy, cx = rng.integers(0, h), rng.integers(0, w)
            ys[b] = np.clip(cy - crop // 2, 0, h - crop)
            xs[b] = np.clip(cx - crop // 2, 0, w - crop)
        tt, yy, xx = (torch.from_numpy(a).to(device) for a in (ti, ys, xs))
        x = dihedral(gather(tiles_t, tt, yy, xx, crop))
        inp = x[:, :n_in] * torch.empty(batch, 1, 1, 1, device=device).uniform_(0.8, 1.25)
        inp = inp + 0.03 * torch.randn_like(inp)
        loss = F.binary_cross_entropy_with_logits(model(inp), x[:, n_in], pos_weight=pos_w, reduction="none")
        loss = (loss * (1 - x[:, n_in + 1])).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return model


def norm(p):
    return p / max(float(np.percentile(p, 99.9)), 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--encoder", type=Path, default=Path("output/encoder.pt"))
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ssl_state = torch.load(a.encoder, map_location="cpu")
    enc = Encoder().to(device)
    enc.load_state_dict(ssl_state)
    data = load_all()
    rng = np.random.default_rng(0)
    totals: dict[str, dict[int, int]] = {}
    targets = 0
    for i, test in enumerate(data[: a.folds]):
        train = [t for j, t in enumerate(data) if j != i]
        targets += test["n"]
        t0 = time.time()
        maps = {"pixel GBM": gbm_prob(train, test, rng)}
        peaks = {
            "pixel GBM": peaks_from_prob(maps["pixel GBM"]),
            "SSL probe": probe_peaks(train, test, enc, device, rng),
        }
        for name, state in (("U-Net scratch", None), ("U-Net from SSL", ssl_state)):
            model = train_unet(train, state, a.steps, device, seed=i)
            maps[name] = predict_dense(model, test["chan"], device)
            peaks[name] = peaks_from_prob(maps[name])
        maps["ensemble GBM+U-Net(SSL)"] = norm(maps["pixel GBM"]) + norm(maps["U-Net from SSL"])
        peaks["ensemble GBM+U-Net(SSL)"] = peaks_from_prob(maps["ensemble GBM+U-Net(SSL)"])
        rows = []
        for name, pk in peaks.items():
            h = hits_at(pk, test)
            for k in BUDGETS:
                totals.setdefault(name, dict.fromkeys(BUDGETS, 0))[k] += h[k]
            rows.append(f"{name}: {h[200]}/{test['n']}")
        print(
            f"fold {i} ({test['bbox'][:2]}, {test['n']} barrows, {time.time() - t0:.0f}s) K=200 -> "
            + "; ".join(rows),
            flush=True,
        )
    print(f"\nPOOLED recall over {targets} held-out barrows (K candidates per tile)")
    print(f"{'method':<28}" + "".join(f"K={k:<6}" for k in BUDGETS))
    for name, per_k in totals.items():
        print(f"{name:<28}" + "".join(f"{per_k[k] / targets:<8.0%}" for k in BUDGETS))


if __name__ == "__main__":
    main()
