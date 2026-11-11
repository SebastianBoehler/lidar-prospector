"""Discovery run: places that look like known barrows but sit outside every scheduled monument.

For each tile the examples are every known barrow in the *other* tiles (so nothing is searched with its own
labels). Output: a contact sheet per tile for visual review and a CSV of candidate coordinates. These are
hypotheses to check against the Historic Environment Record, not discoveries.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage

from .crossval import tiles
from .embed import Encoder, embed_grid, relief_channels
from .labels import monuments
from .learn import rasters
from .retrieve import Index
from .terrain import hillshade
from .wcs import fetch_dtm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CROP = 48  # half-width in metres of each thumbnail


def load(bbox, model, device):
    dtm, x0, y1 = fetch_dtm(*bbox)
    mons = monuments(*bbox)
    tight, ids, *_ = rasters(dtm, x0, y1, mons)
    shapes = [({"type": "Polygon", "coordinates": m["rings"]}, 1) for m in mons]
    covered = rasterize(shapes, out_shape=dtm.shape, transform=from_origin(x0, y1, 1, 1), dtype="uint8") > 0
    n = len(tight)
    centres = np.array(ndimage.center_of_mass(ids > 0, ids, range(1, n + 1))) if n else np.zeros((0, 2))
    return {
        "bbox": bbox, "x0": x0, "y1": y1, "shade": hillshade(dtm),
        "index": Index(embed_grid(model, relief_channels(dtm), device=device)),
        "near_known": ndimage.binary_dilation(covered, iterations=30), "covered": covered, "centres": centres,
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=Path, default=Path("output/encoder.pt"))
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--out", type=Path, default=Path("output"))
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Encoder().to(device)
    model.load_state_dict(torch.load(a.encoder, map_location=device))
    data = [load(b, model, device) for b in tiles()]
    rows = []
    for ti, t in enumerate(data):
        vecs = [
            o["index"].flat[o["index"].cell_of(*c)]
            for oi, o in enumerate(data)
            if oi != ti
            for c in o["centres"]
        ]
        idx = t["index"]
        scores = idx.probe_scores(np.stack(vecs), None, np.random.default_rng(ti))
        top = idx.top(scores, 400)
        on_known = sum(bool(t["covered"][idx.pixel_of(c)]) for c in top[:100])
        fresh = [c for c in top if not t["near_known"][idx.pixel_of(c)]][: a.top]
        print(f"tile {t['bbox'][:2]}: top-100 by similarity: {on_known} fall inside a scheduled monument; "
              f"{len(fresh)} unscheduled candidates kept")  # fmt: skip
        fig, axes = plt.subplots(4, 6, figsize=(15, 10.5))
        for ax in axes.ravel():
            ax.axis("off")
        for rank, (ax, cell) in enumerate(zip(axes.ravel(), fresh, strict=False), 1):
            r, c = idx.pixel_of(cell)
            crop = t["shade"][max(r - CROP, 0) : r + CROP, max(c - CROP, 0) : c + CROP]
            ax.imshow(crop, cmap="gray", vmin=0.3, vmax=1)
            ax.set_title(f"#{rank}  sim {scores[cell]:.2f}", fontsize=9)
            rows.append([ti, t["x0"] + c, t["y1"] - r, round(float(scores[cell]), 3)])
        fig.suptitle(
            f"tile {t['bbox'][0]},{t['bbox'][1]}: look-alikes of known barrows outside scheduled areas"
        )
        fig.tight_layout()
        fig.savefig(a.out / f"discovery_{t['bbox'][0]}_{t['bbox'][1]}.png", dpi=70)
        plt.close(fig)
    with open(a.out / "discovery_candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "easting", "northing", "similarity"])
        w.writerows(rows)
    print(f"wrote {len(rows)} candidates")


if __name__ == "__main__":
    main()
