"""Labelled pretraining on open typed heritage polygons, as the honest baseline for the label-free route.

The corpus tiles on disk (disjoint from the 13 benchmark tiles) are labelled from Historic England's scheduled
monuments. The same U-Net that dense-contrastive pretraining produces is trained here with labels, then its
decoder features go through the frozen prompt benchmark unchanged. Two variants:

  typed : positives are the single barrows / cairns / hut circles / stone settings (the benchmark's types),
          other monuments ignored. Supervised, but on different tiles from the benchmark.
  other : positives are every scheduled monument whose name is NOT one of those types (enclosures, field
          systems, dykes, ...); the benchmark's types are ignored, never a positive or a negative. Tests
          whether knowing *monuments in general* transfers to prompts for types the network never saw.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rasterio import Affine
from rasterio.features import rasterize
from scipy import ndimage

from .embed import dihedral, relief_channels
from .features_torch import pick_device
from .labels import BARROW, monuments
from .prompt_types import object_type
from .unet import UNet
from .wcs import fetch_dtm

HELD = re.compile(r"hut circle|hut-circle|roundhouse|stone row|stone circle|standing stone|menhir", re.I)
CROP = 128


def is_benchmark_type(m: dict) -> bool:
    name = m["name"] or ""
    typed = object_type(name) not in (None, "small enclosure") and (m["area_ha"] or 9) <= 0.5
    return typed or bool(BARROW.search(name) or HELD.search(name))


def label_maps(bbox, mons, mode: str, shape=(3000, 3000)) -> np.ndarray:
    """(2, H, W) uint8: [positive, ignore] for one tile (rows run north to south from the bbox top)."""
    tf = Affine(1, 0, bbox[0], 0, -1, bbox[3])

    def burn(ms):
        if not ms:
            return np.zeros(shape, bool)
        g = [({"type": "Polygon", "coordinates": m["rings"]}, 1) for m in ms]
        return rasterize(g, out_shape=shape, transform=tf, dtype="uint8") > 0

    bench = [m for m in mons if is_benchmark_type(m)]
    if mode == "typed":
        pos_ms = [m for m in bench if object_type(m["name"]) not in (None, "small enclosure")]
        pos = ndimage.binary_dilation(burn(pos_ms), iterations=2)
        ign = ndimage.binary_dilation(burn([m for m in mons if m not in pos_ms]), iterations=3) & ~pos
    else:
        pos = burn([m for m in mons if not is_benchmark_type(m)])
        ign = ndimage.binary_dilation(burn(bench), iterations=3) & ~pos
    return np.stack([pos, ign]).astype(np.uint8)


def build(mode: str, root: Path = Path("data")):
    """(relief memmaps, label memmaps) for every labelled-corpus English tile; label caches are reused."""
    manifest = json.loads((root / "corpus_manifest.json").read_text())
    relief, labels = [], []
    for rec in manifest:
        if rec["source"] != "ea" or rec["valid"] < 0.5:
            continue
        b = tuple(rec["bbox"])
        lab = root / "labels" / f"{mode}_{b[0]}_{b[1]}.npy"
        if not lab.exists():
            raw = root / "labels" / f"ea_{b[0]}_{b[1]}.json"
            if not raw.exists():
                raw.write_text(json.dumps(monuments(*b)))
            np.save(lab, label_maps(b, json.loads(raw.read_text()), mode))
        rel = root / "relief" / f"ea_{b[0]}_{b[1]}.npy"
        if not rel.exists():
            np.save(rel, relief_channels(fetch_dtm(*b)[0]).astype(np.float16))
        relief.append(np.load(rel, mmap_mode="r"))
        labels.append(np.load(lab, mmap_mode="r"))
    return relief, labels


def sampler(relief, labels, rng, batch: int):
    """Half the crops centred near a positive pixel (tiles weighted by how many they hold), half uniform."""
    pos = [np.argwhere(np.asarray(m[0, ::4, ::4])) * 4 for m in labels]
    weight = np.array([len(p) for p in pos], float)
    weight = weight / weight.sum() if weight.sum() else None
    h, w = relief[0].shape[1:]
    out = np.empty((batch, 4, CROP, CROP), np.float32)
    for i in range(batch):
        if i % 2 == 0 and weight is not None:
            t = rng.choice(len(pos), p=weight)
            cy, cx = pos[t][rng.integers(len(pos[t]))] + rng.integers(-48, 49, 2)
        else:
            t, cy, cx = rng.integers(len(relief)), rng.integers(h), rng.integers(w)
        y = int(np.clip(cy - CROP // 2, 0, h - CROP))
        x = int(np.clip(cx - CROP // 2, 0, w - CROP))
        out[i, :2] = relief[t][:, y : y + CROP, x : x + CROP]
        out[i, 2:] = labels[t][:, y : y + CROP, x : x + CROP]
    return out


def train(relief, labels, steps: int, device: str, seed: int = 0, batch: int = 24, pos_weight: float = 10.0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    pw = torch.tensor(pos_weight, device=device)
    for step in range(steps):
        x = dihedral(torch.from_numpy(sampler(relief, labels, rng, batch)).to(device))
        scale = torch.empty(batch, 1, 1, 1, device=device).uniform_(0.8, 1.25)
        inp = x[:, :2] * scale + 0.03 * torch.randn_like(x[:, :2])
        loss = F.binary_cross_entropy_with_logits(model(inp), x[:, 2], pos_weight=pw, reduction="none")
        loss = (loss * (1 - x[:, 3])).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == steps - 1:
            print(f"step {step + 1:5d}/{steps}  loss {loss.item():.4f}", flush=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["typed", "other"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    relief, labels = build(a.mode)
    n_pos = sum(int(np.asarray(m[0, ::4, ::4]).sum()) for m in labels)
    print(f"{len(relief)} tiles; {n_pos} positive pixels (1/16 subsampled) in mode '{a.mode}'", flush=True)
    model = train(relief, labels, a.steps, pick_device())
    out = a.out or Path(f"output/weak_{a.mode}.pt")
    out.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
