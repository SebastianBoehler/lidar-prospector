"""Pick and download label-dense English 3 km cells for supervised pretraining, disjoint from the benchmark.

Cells are chosen by how many single barrows, cairns, hut circles and stone settings Historic England
schedules in them (rarer types weigh more), capped per region for variety. They are added to the corpus
manifest (region "label-dense") so the same relief caches and label rasterisers pick them up. Resumable.
"""

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

from .corpus import CELL, MANIFEST, fetch
from .labels import URL, _session
from .prompt_types import object_type

REGIONS = {
    "Dartmoor": (250000, 60000, 290000, 100000),
    "Exmoor": (270000, 120000, 320000, 150000),
    "Bodmin and West Penwith": (160000, 20000, 240000, 90000),
    "Salisbury Plain and Marlborough": (380000, 120000, 430000, 190000),
    "Dorset downs": (350000, 80000, 400000, 115000),
    "North York Moors": (440000, 480000, 480000, 520000),
    "Peak District": (400000, 360000, 440000, 410000),
    "Yorkshire Dales": (370000, 460000, 420000, 500000),
    "Northumberland": (380000, 590000, 430000, 640000),
    "Lake District": (310000, 480000, 350000, 520000),
    "Yorkshire Wolds": (480000, 440000, 520000, 480000),
    "Shropshire hills": (330000, 270000, 370000, 320000),
}
WEIGHT = {"round barrow": 1, "cairn": 2, "hut circle": 4, "stone setting": 4}
BLOCK = 12000


def block_monuments(x0, y0):
    """(cell, type) for every small typed monument whose centroid falls in the 12 km block."""
    s = _session()
    out, offset = [], 0
    geom = {
        "xmin": x0,
        "ymin": y0,
        "xmax": x0 + BLOCK,
        "ymax": y0 + BLOCK,
        "spatialReference": {"wkid": 27700},
    }
    while True:
        r = s.get(
            URL,
            params={"where": "1=1", "geometryType": "esriGeometryEnvelope", "inSR": 27700, "outSR": 27700,
                    "geometry": json.dumps(geom), "spatialRel": "esriSpatialRelIntersects",
                    "outFields": "Name,area_ha", "returnGeometry": "false", "returnCentroid": "true",
                    "resultOffset": offset, "f": "json"},
            timeout=60,
        )  # fmt: skip
        r.raise_for_status()
        body = r.json()
        for f in body["features"]:
            a = f["attributes"]
            t = object_type(a["Name"])
            if t in WEIGHT and a["area_ha"] is not None and a["area_ha"] <= 0.5:
                c = f["centroid"]
                out.append(((int(c["x"]) // CELL * CELL, int(c["y"]) // CELL * CELL), t))
        if not body.get("exceededTransferLimit"):
            return out
        offset += len(body["features"])


def choose(n: int, per_region: int, avoid) -> list[dict]:
    jobs = [
        (name, x, y)
        for name, (a, b, c, d) in REGIONS.items()
        for x in range(a // BLOCK * BLOCK, c, BLOCK)
        for y in range(b // BLOCK * BLOCK, d, BLOCK)
    ]
    with ThreadPoolExecutor(4) as ex:
        found = list(ex.map(lambda j: block_monuments(j[1], j[2]), jobs))
    counts: dict[tuple, Counter] = defaultdict(Counter)
    region_of = {}
    for (name, _, _), res in zip(jobs, found, strict=True):
        for cell, t in res:
            counts[cell][t] += 1
            region_of[cell] = name
    ranked = sorted(counts, key=lambda c: -sum(WEIGHT[t] * k for t, k in counts[c].items()))
    taken, chosen, per = list(avoid), [], Counter()
    for cell in ranked:
        if len(chosen) == n:
            break
        x, y = cell
        if per[region_of[cell]] >= per_region:
            continue
        if any(abs(x - tx) < CELL and abs(y - ty) < CELL for tx, ty in taken):
            continue
        taken.append(cell)
        per[region_of[cell]] += 1
        chosen.append({"source": "ea", "region": "label-dense", "area": region_of[cell],
                       "bbox": (x, y, x + CELL, y + CELL), "types": dict(counts[cell])})  # fmt: skip
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--per-region", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from .benchmark_nl import CELLS
    from .crossval import landscape_tiles, tiles

    done = json.loads(MANIFEST.read_text())
    avoid = [b[:2] for b in list(tiles()) + [b for _, b in landscape_tiles()]] + list(CELLS)
    avoid += [tuple(d["bbox"][:2]) for d in done]
    items = choose(a.n, a.per_region, avoid)
    total = Counter()
    for i in items:
        total.update(i["types"])
        print(f"{i['area']:<32}{i['bbox'][:2]}  {i['types']}")
    print(f"{len(items)} cells (~{len(items) * 37.7 / 1000:.1f} GB); labelled objects: {dict(total)}")
    if a.dry_run:
        return
    have = {tuple(d["bbox"]) for d in done}
    for item in items:
        if tuple(item["bbox"]) in have:
            continue
        try:
            rec = fetch(item)
        except (requests.RequestException, RuntimeError, OSError) as e:
            print(f"FAILED {item['bbox']}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        done.append({k: v for k, v in rec.items() if k != "types"})
        MANIFEST.write_text(json.dumps(done, indent=1))
        print(f"[{len(done)}] {item['area']} {item['bbox'][:2]} valid {rec['valid']:.0%}", flush=True)


if __name__ == "__main__":
    main()
