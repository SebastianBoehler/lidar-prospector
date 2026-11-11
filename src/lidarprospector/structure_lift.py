"""How well do the line detectors cover linear monuments, enclosures and field systems?

Labels are whole protected areas (a dyke's polygon is its corridor, a hillfort's is its interior and
ramparts), so this is a coverage check, not a precise recall. For each class and pyramid level:
  lift      = share of detected line length inside the class zone / share of the area that is zone
  coverage  = share of the zone within 10 m of a detected line, vs the same share over the whole tile
Roads are not masked, so modern roads that cross a zone count as hits.
"""

import argparse
from dataclasses import replace

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage

from .crossval import landscape_tiles, tiles
from .labels import monuments
from .pipeline import ScanConfig, scan
from .runtime import MemoryGate, make_plan
from .sources import GeoTiffSource
from .tiling import required_halo

CLASSES = ("linear", "enclosure", "fields")
SCALES = (1, 2, 4, 8)


def tile_stats(bbox, stats):
    path = f"data/ea_dtm_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.tif"
    src = GeoTiffSource(path, "OGL v3", "(c) Environment Agency")
    plan = replace(make_plan(required_halo(SCALES)), tile_core=1024)
    res = scan(
        src, bbox, plan, ScanConfig(scales=SCALES, min_extent_m=80.0), log=lambda *_: None, gate=MemoryGate(0)
    )
    shape = (round(bbox[3] - bbox[1]), round(bbox[2] - bbox[0]))
    tf = from_origin(bbox[0], bbox[3], 1, 1)
    mons = monuments(*bbox)
    line_mask = {s: np.zeros(shape, bool) for s in SCALES}
    length = dict.fromkeys(SCALES, 0.0)
    pts = []  # (scale, rows, cols) per line, base pixels
    for line, s in res.lines:
        r = np.clip(line.pixels[:, 0] * s + s // 2, 0, shape[0] - 1)
        c = np.clip(line.pixels[:, 1] * s + s // 2, 0, shape[1] - 1)
        line_mask[s][r, c] = True
        length[s] += len(r) * s
        pts.append((s, r, c))
    near = {s: ndimage.binary_dilation(line_mask[s], iterations=10) for s in SCALES}
    for cls in CLASSES:
        polys = [
            ({"type": "Polygon", "coordinates": m["rings"]}, 1)
            for m in mons
            if m["cls"] == cls and m["area_ha"] and m["area_ha"] > 0.5
        ]
        if not polys:
            continue
        zone = ndimage.binary_dilation(
            rasterize(polys, out_shape=shape, transform=tf, dtype="uint8") > 0, iterations=20
        )
        for s in SCALES:
            acc = stats.setdefault((cls, s), np.zeros(6))
            inside = sum(int(zone[r, c].sum()) * s for sc, r, c in pts if sc == s)
            acc += [inside, length[s], zone.sum(), zone.size, (near[s] & zone).sum(), near[s].sum()]
    return len(mons)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    boxes = list(tiles()) + [b for _, b in landscape_tiles()]
    stats: dict = {}
    for b in boxes:
        tile_stats(b, stats)
        print(f"done {b[:2]}", flush=True)
    print("\nclass       scale  zones(km2)  lift   zone covered by a line  (tile-wide base rate)")
    for (cls, s), (inside, total, zone, area, zone_near, all_near) in sorted(stats.items()):
        if zone == 0 or total == 0:
            continue
        print(f"{cls:<10} {s:>4} m {zone / 1e6:9.2f} {(inside / total) / (zone / area):6.1f}x "
              f"{zone_near / zone:20.0%} {all_near / area:20.0%}")  # fmt: skip


if __name__ == "__main__":
    main()
