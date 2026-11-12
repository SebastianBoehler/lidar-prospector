"""Dense contrastive pretraining on every cached terrain tile (no labels)."""

import argparse
from pathlib import Path

import torch

from .dense_contrast import pretrain
from .features_torch import pick_device
from .train_mae import corpus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("output/dense.pt"))
    ap.add_argument("--every", type=int, default=1, help="use every n-th terrain (a data-scaling probe)")
    ap.add_argument("--big", action="store_true", help="train from the on-disk corpus (data/relief)")
    a = ap.parse_args()
    if a.big:
        from .corpus import build_store

        dtms = build_store()
        print(f"{len(dtms)} terrains on disk, no labels", flush=True)
    else:
        dtms = corpus()[:: a.every]
        print(f"{len(dtms)} terrains, {sum(d.size for d in dtms) / 1e6:.0f} km2, no labels", flush=True)
    net, sps = pretrain(dtms, steps=a.steps, device=pick_device())
    a.out.parent.mkdir(exist_ok=True)
    torch.save(net.state_dict(), a.out)
    print(f"saved {a.out}; throughput {sps:.0f} crops/s of 128x128 m (two views each)")


if __name__ == "__main__":
    main()
