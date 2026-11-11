"""Score the detector against scheduled barrows: recall, and candidate density elsewhere."""

import argparse
import re

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage

from .detect import detect
from .labels import barrows
from .wcs import fetch_dtm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=int, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"), required=True)
    ap.add_argument(
        "--tol-m", type=float, default=10.0, help="candidate within this of a polygon counts as a hit"
    )
    a = ap.parse_args()

    dtm, x0, y1 = fetch_dtm(*a.bbox)
    labels = barrows(*a.bbox)
    cands, _, _ = detect(dtm)

    shapes = [({"type": "Polygon", "coordinates": l["rings"]}, i + 1) for i, l in enumerate(labels)]
    ids = rasterize(shapes, out_shape=dtm.shape, transform=from_origin(x0, y1, 1, 1), dtype="int32")
    dist, (ri, ci) = ndimage.distance_transform_edt(ids == 0, return_indices=True)
    near = np.where(dist <= a.tol_m, ids[ri, ci], 0)

    hit_ids, matched = set(), 0
    for c in cands:
        k = near[min(int(c.row), dtm.shape[0] - 1), min(int(c.col), dtm.shape[1] - 1)]
        if k:
            hit_ids.add(k)
            matched += 1
    area_km2 = np.sum(~np.isnan(dtm)) / 1e6
    n = len(labels)
    print(f"window {area_km2:.2f} km2 with data; {n} single-barrow labels; {len(cands)} candidates")
    print(
        f"recall: {len(hit_ids)}/{n} = {len(hit_ids) / max(n, 1):.0%} "
        f"of barrows have a candidate within {a.tol_m:.0f} m"
    )
    print(
        f"candidates on labels: {matched}; elsewhere: {len(cands) - matched} "
        f"({(len(cands) - matched) / area_km2:.1f}/km2, unlabelled does not mean false)"
    )
    rng = np.random.default_rng(0)
    valid = np.argwhere(~np.isnan(dtm))
    rand = []
    for _ in range(50):
        pts = valid[rng.integers(0, len(valid), len(cands))]
        rand.append(len(set(near[pts[:, 0], pts[:, 1]]) - {0}) / max(n, 1))
    print(
        f"random baseline (same number of random points): recall {np.mean(rand):.1%} +/- {np.std(rand):.1%}"
    )

    kinds = {}
    for i, l in enumerate(labels, 1):
        m = re.search(r"\b(bowl|bell|disc|pond|long|saucer|round)\b", l["name"], re.I)
        k = m.group(1).lower() if m else "other"
        h, t = kinds.get(k, (0, 0))
        kinds[k] = (h + (i in hit_ids), t + 1)
    print(
        "recall by type: "
        + ", ".join(f"{k} {h}/{t}" for k, (h, t) in sorted(kinds.items(), key=lambda kv: -kv[1][1]))
    )


if __name__ == "__main__":
    main()
