"""Wall-clock cost of each pipeline stage on one cached tile, as throughput."""

import argparse
import time

import numpy as np

from .crossval import STONEHENGE
from .detect import detect
from .features import feature_stack
from .shapes import depressions, linear_features
from .terrain import hillshade, local_relief
from .wcs import fetch_dtm


def timed(label, fn, km2, rows):
    t = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t
    rows.append((label, dt, km2 / dt))
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    rows: list[tuple[str, float, float]] = []
    dtm, _, _ = timed("read cached GeoTIFF", lambda: fetch_dtm(*STONEHENGE), 9.0, rows)
    km2 = np.sum(~np.isnan(dtm)) / 1e6
    timed("local relief + hillshade", lambda: (local_relief(dtm), hillshade(dtm)), km2, rows)
    timed("feature stack (13 channels)", lambda: feature_stack(dtm), km2, rows)
    timed("hand-tuned mound detector", lambda: detect(dtm), km2, rows)
    timed("depression finder", lambda: depressions(dtm), km2, rows)
    timed("linear earthwork finder", lambda: linear_features(dtm), km2, rows)
    try:
        import torch

        from .embed import Encoder, embed_grid, relief_channels

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = Encoder().to(device)
        chan = relief_channels(dtm)
        embed_grid(model, chan[:, :256, :256], device=device)  # warm-up
        timed("relief channels (CPU)", lambda: relief_channels(dtm), km2, rows)
        timed(f"encoder over tile ({device})", lambda: embed_grid(model, chan, device=device), km2, rows)
    except ImportError:
        print("torch not installed: skipping encoder stages")
    print(f"{'stage':<32}{'seconds':>9}{'km2/s':>9}")
    for label, dt, rate in rows:
        print(f"{label:<32}{dt:9.2f}{rate:9.2f}")


if __name__ == "__main__":
    main()
