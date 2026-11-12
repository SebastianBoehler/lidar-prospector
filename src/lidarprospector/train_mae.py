"""Pretrain the masked-terrain network on every cached terrain tile (no labels)."""

import argparse
from pathlib import Path

import torch

from .benchmark_nl import CELLS
from .crossval import landscape_tiles, tiles
from .features_torch import pick_device
from .mae import pretrain
from .sources import GeoTiffSource
from .sources.ahn import NetherlandsAHN
from .wcs import fetch_dtm


def corpus():
    """All terrain on disk: the 13 English tiles, 4 Dutch tiles and the Guatemala survey, at 1 m."""
    dtms = [fetch_dtm(*b)[0] for b in list(tiles()) + [b for _, b in landscape_tiles()]]
    ahn = NetherlandsAHN()
    dtms += [ahn.fetch((x, y, x + 3000, y + 3000), cell=1.0).z for x, y in CELLS]
    q = Path("data/quirigua/quirigua_museo_dtm_cog.tif")
    if q.exists():
        src = GeoTiffSource(str(q), "CC BY 4.0", "Maurer 2026")
        b = (280558, 1688608, 281313, 1689943)
        dtms.append(src.fetch(b, cell=1.0).z)
    return dtms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--block", type=int, default=16, help="side of each hidden block, in metres")
    ap.add_argument("--out", type=Path, default=Path("output/mae.pt"))
    a = ap.parse_args()
    dtms = corpus()
    km2 = sum(d.size for d in dtms) / 1e6
    print(f"{len(dtms)} terrains, {km2:.0f} km2, no labels", flush=True)
    net, sps = pretrain(dtms, steps=a.steps, device=pick_device(), block=a.block)
    a.out.parent.mkdir(exist_ok=True)
    torch.save(net.state_dict(), a.out)
    print(f"saved {a.out}; throughput {sps:.0f} crops/s of 128x128 m")


if __name__ == "__main__":
    main()
