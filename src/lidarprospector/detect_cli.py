import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .detect import detect
from .terrain import hillshade


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtm", type=Path, default=Path("output/dtm.npy"))
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    dtm = np.load(a.dtm)
    cands, lrm, _ = detect(dtm)
    print(
        f"{len(cands)} candidates ({sum(c.kind == 'mound' for c in cands)} mound-like, "
        f"{sum(c.kind == 'ridge' for c in cands)} ridge-like)"
    )
    for c in cands[: a.top]:
        print(
            f"#{c.id:<3} {c.kind:5} row {c.row:6.0f} col {c.col:6.0f}  area {c.area_m2:7.0f} m2  "
            f"peak {c.peak_m:4.2f} m  elong {c.elongation:4.1f}  "
            f"solid {c.solidity:4.2f}  score {c.score:5.1f}"
        )
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(hillshade(dtm), cmap="gray")
    for rank, c in enumerate(cands[: a.top]):
        ax.plot(
            c.col,
            c.row,
            "o" if c.kind == "mound" else "s",
            mfc="none",
            mec="red" if c.kind == "mound" else "cyan",
            ms=14,
            mew=1.6,
        )
        ax.text(c.col + 12, c.row, str(rank), color="yellow", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(a.dtm.parent / "candidates.png", dpi=100)


if __name__ == "__main__":
    main()
