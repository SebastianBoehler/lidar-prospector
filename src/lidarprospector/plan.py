"""Dry run: report how much data a window needs without downloading any point data."""

import argparse
from collections import Counter

from .ept import EptProject, window_3857


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--half-m", type=float, default=1000)
    a = ap.parse_args()

    proj = EptProject(a.project)
    win = window_3857(a.lat, a.lon, a.half_m)
    nodes = proj.nodes(win)
    per_depth = Counter()
    for n in nodes:
        per_depth[n.depth] += n.points
    total = sum(n.points for n in nodes)
    size = proj.sizes(nodes)
    area_km2 = (2 * a.half_m / 1000) ** 2
    print(
        f"{a.project}: {len(nodes)} nodes, {total / 1e6:.1f}M points (node-level, incl. margin), "
        f"{size / 1e6:.0f} MB compressed, window {area_km2:.1f} km2"
    )
    for d in sorted(per_depth):
        print(f"  depth {d}: {per_depth[d] / 1e6:.2f}M")


if __name__ == "__main__":
    main()
