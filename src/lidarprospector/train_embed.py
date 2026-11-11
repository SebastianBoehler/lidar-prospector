"""Train the self-supervised terrain encoder on the cached benchmark tiles (no labels used)."""

import argparse
from pathlib import Path

import torch

from .crossval import tiles
from .embed import train
from .wcs import fetch_dtm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("output/encoder.pt"))
    a = ap.parse_args()
    dtms = [fetch_dtm(*b)[0] for b in tiles()]
    model = train(dtms, steps=a.steps, batch=a.batch)
    a.out.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
