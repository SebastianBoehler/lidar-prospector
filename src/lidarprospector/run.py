"""Stream a window, build bare earth, render terrain products."""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ept import EptProject, window_3857
from .terrain import ground_dtm, hillshade, local_relief


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--half-m", type=float, default=500)
    ap.add_argument("--utm-epsg", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("output"))
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True)

    proj = EptProject(a.project)
    win = window_3857(a.lat, a.lon, a.half_m)
    nodes = proj.nodes(win)

    t = time.time()
    x, y, z, cls = proj.fetch(nodes, win, a.utm_epsg)
    print(f"fetched {len(x) / 1e6:.1f}M points in {time.time() - t:.0f}s; ground {np.mean(cls == 2):.0%}")

    t = time.time()
    dtm, _ = ground_dtm(x, y, z, cls)
    lrm = local_relief(dtm)
    hs = hillshade(dtm)
    print(
        f"terrain {dtm.shape} in {time.time() - t:.1f}s; relief range "
        f"{np.nanpercentile(lrm, 1):.2f}..{np.nanpercentile(lrm, 99):.2f} m"
    )
    np.save(a.out / "dtm.npy", dtm)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(hs, cmap="gray")
    ax[0].set_title("multi-direction hillshade")
    ax[1].imshow(lrm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[1].set_title("local relief (m)")
    for p in ax:
        p.axis("off")
    fig.tight_layout()
    fig.savefig(a.out / "terrain.png", dpi=110)


if __name__ == "__main__":
    main()
