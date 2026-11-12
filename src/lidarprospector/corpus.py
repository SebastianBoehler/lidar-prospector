"""A larger, varied terrain corpus for label-free pretraining, with a provenance manifest.

Cells are sampled (seeded, reproducible) from named regions chosen for variety: upland, forest, flat arable,
coast, chalk, fen. Each downloaded tile is recorded in `data/corpus_manifest.json` with its source, licence,
bounds, size and how much of it holds data. Downloads resume: cached tiles are skipped.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .sources import EnglandEA
from .sources.ahn import NetherlandsAHN

CELL = 3000
# name -> (xmin, ymin, xmax, ymax) in the source's grid (EPSG:27700 England, EPSG:28992 Netherlands)
EA_REGIONS = {
    "Cotswolds": (390000, 200000, 420000, 230000),
    "Wessex downs": (400000, 120000, 440000, 160000),
    "Yorkshire Dales": (380000, 460000, 410000, 490000),
    "Lake District": (320000, 490000, 350000, 520000),
    "Lincolnshire fens": (520000, 290000, 560000, 330000),
    "Breckland heath and forest": (570000, 270000, 600000, 300000),
    "Kent Downs and Weald": (560000, 130000, 600000, 160000),
    "Cornwall moor": (170000, 30000, 230000, 80000),
    "South Pennines": (400000, 400000, 440000, 440000),
    "Northumberland uplands": (380000, 600000, 420000, 640000),
    "Chilterns and Thames": (470000, 180000, 510000, 220000),
    "Welsh border woodland": (350000, 200000, 390000, 240000),
    "Exmoor": (280000, 130000, 320000, 150000),
    "New Forest": (420000, 90000, 450000, 110000),
    "Midlands arable": (430000, 270000, 470000, 310000),
}
NL_REGIONS = {
    "Veluwe": (170000, 450000, 200000, 480000),
    "Drenthe heath": (230000, 540000, 260000, 570000),
    "Flevoland polders": (150000, 480000, 180000, 510000),
    "Holland dunes": (90000, 470000, 110000, 500000),
    "South Limburg hills": (180000, 310000, 200000, 330000),
    "Betuwe rivers": (160000, 430000, 190000, 445000),
    "Friesland and Groningen": (190000, 580000, 220000, 600000),
}
MANIFEST = Path("data/corpus_manifest.json")


def plan(seed: int = 0, ea_per_region: int = 7, nl_per_region: int = 6, avoid=()) -> list[dict]:
    rng = np.random.default_rng(seed)
    taken = {tuple(a) for a in avoid}
    items = []
    for source, regions, n in (("ea", EA_REGIONS, ea_per_region), ("ahn", NL_REGIONS, nl_per_region)):
        for name, (x0, y0, x1, y1) in regions.items():
            gx, gy = np.arange(x0 // CELL, x1 // CELL), np.arange(y0 // CELL, y1 // CELL)
            cells = [(int(x) * CELL, int(y) * CELL) for x in gx for y in gy]
            rng.shuffle(cells)
            got = 0
            for x, y in cells:
                if got == n:
                    break
                if all(abs(x - tx) >= CELL or abs(y - ty) >= CELL for tx, ty in taken):
                    taken.add((x, y))
                    items.append({"source": source, "region": name, "bbox": (x, y, x + CELL, y + CELL)})
                    got += 1
    return items


def fetch(item: dict) -> dict:
    bbox = item["bbox"]
    src = EnglandEA() if item["source"] == "ea" else NetherlandsAHN()
    dem = src.fetch(bbox, cell=1.0)
    nbytes = sum(
        p.stat().st_size for p in Path("data").glob(f"*{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}*.tif")
    )
    return {**item, "bbox": list(bbox), "licence": src.licence, "attribution": src.attribution,
            "bytes": nbytes, "valid": round(float(np.mean(~np.isnan(dem.z))), 3)}  # fmt: skip


def download(items: list[dict], workers: int = 3, log=print) -> list[dict]:
    done = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    have = {tuple(d["bbox"]) for d in done}
    todo = [i for i in items if tuple(i["bbox"]) not in have]
    lock = threading.Lock()
    pools = {"ea": ThreadPoolExecutor(workers), "ahn": ThreadPoolExecutor(workers)}
    log(f"{len(todo)} tiles to fetch ({len(items) - len(todo)} already in the manifest)")

    def run(item):
        try:
            rec = fetch(item)
        except Exception as e:  # one bad cell (coast, outage) must not stop the corpus
            log(f"FAILED {item['source']} {item['bbox']}: {type(e).__name__}: {str(e)[:120]}")
            return
        with lock:
            done.append(rec)
            MANIFEST.write_text(json.dumps(done, indent=1))
            gb = sum(d["bytes"] for d in done) / 1e9
            where = f"{rec['source']} {rec['region']} {rec['bbox'][:2]}"
            log(f"[{len(done)}] {where} valid {rec['valid']:.0%}  total {gb:.2f} GB")

    futures = [pools[i["source"]].submit(run, i) for i in todo]
    for f in futures:
        f.result()
    for p in pools.values():
        p.shutdown()
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=3, help="parallel downloads per source")
    a = ap.parse_args()
    from .benchmark_nl import CELLS
    from .crossval import landscape_tiles, tiles

    avoid = [b[:2] for b in list(tiles()) + [b for _, b in landscape_tiles()]] + list(CELLS)
    items = plan(avoid=avoid)
    ea = sum(i["source"] == "ea" for i in items)
    nl = len(items) - ea
    print(
        f"{len(items)} tiles: {ea} England (~{ea * 37.7 / 1000:.1f} GB), {nl} Netherlands "
        f"(~{nl * 22.9 / 1000:.1f} GB); {len(items) * 9} km2"
    )
    if not a.dry_run:
        download(items, a.workers)


if __name__ == "__main__":
    main()


def build_store(cache: Path = Path("data/relief"), min_valid: float = 0.5):
    """CropStore over everything on disk: the 18 benchmark-era terrains plus the downloaded manifest tiles
    that hold enough data. Relief channels are cached once; later calls are instant."""
    from .benchmark_nl import CELLS
    from .crop_store import CropStore
    from .crossval import landscape_tiles, tiles
    from .sources import GeoTiffSource
    from .wcs import fetch_dtm

    ahn = NetherlandsAHN()
    items = []
    for b in list(tiles()) + [b for _, b in landscape_tiles()]:
        items.append((f"ea_{b[0]}_{b[1]}", lambda b=b: fetch_dtm(*b)[0]))
    for x, y in CELLS:
        items.append((f"ahn_{x}_{y}", lambda x=x, y=y: ahn.fetch((x, y, x + CELL, y + CELL), cell=1.0).z))
    q = Path("data/quirigua/quirigua_museo_dtm_cog.tif")
    if q.exists():
        src = GeoTiffSource(str(q), "CC BY 4.0", "Maurer 2026")
        items.append(("quirigua", lambda: src.fetch((280558, 1688608, 281313, 1689943), cell=1.0).z))
    for rec in json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []:
        if rec["valid"] < min_valid:
            continue
        b = tuple(rec["bbox"])
        if rec["source"] == "ea":
            items.append((f"ea_{b[0]}_{b[1]}", lambda b=b: fetch_dtm(*b)[0]))
        else:
            items.append((f"ahn_{b[0]}_{b[1]}", lambda b=b: ahn.fetch(b, cell=1.0).z))
    seen, unique = set(), []
    for key, load in items:  # benchmark tiles can also appear in the manifest: keep one copy
        if key not in seen:
            seen.add(key)
            unique.append((key, load))
    return CropStore.build(unique, cache)
