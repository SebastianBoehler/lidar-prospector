"""Train the mound U-Net on every benchmark tile and save the weights for `scan --mounds`."""

import argparse
from pathlib import Path

import torch

from .features_torch import pick_device
from .models_bench import load_all, train_unet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--init", choices=["scratch", "ssl"], default="scratch")
    ap.add_argument("--encoder", type=Path, default=Path("output/encoder.pt"))
    ap.add_argument("--out", type=Path, default=Path("output/unet.pt"))
    a = ap.parse_args()
    device = pick_device()
    state = torch.load(a.encoder, map_location="cpu") if a.init == "ssl" else None
    model = train_unet(load_all(), state, a.steps, device, seed=0)
    a.out.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
