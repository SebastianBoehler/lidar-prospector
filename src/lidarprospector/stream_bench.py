"""Download-then-process versus streaming, on a simulated slow home connection.

A local range server (rangeserver) serves a real cloud-optimised GeoTIFF with a shared bandwidth cap and
per-request latency; every strategy reads it over real HTTP range requests and the server counts the bytes.
"""

import argparse
import itertools
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import rasterio

from .mounds import unet_mound_fn
from .pipeline import ScanConfig, scan
from .rangeserver import serve
from .runtime import MemoryGate, make_plan
from .sources import GeoTiffSource
from .tiling import required_halo

DEFAULT_COG = "data/quirigua/quirigua_museo_dtm_cog.tif"


def run(label, meter, base, uid, cfg, workers, bbox, core=1024, scales=(1,)):
    meter.reset()
    src = GeoTiffSource(f"{base}/{next(uid)}.tif", "CC BY 4.0", "Maurer 2026")
    plan = replace(make_plan(required_halo(scales, cfg.mound_scales), io_workers=workers), tile_core=core)
    marks = []
    t0 = time.time()
    res = scan(
        src, bbox, plan, cfg, log=lambda *_: None, gate=MemoryGate(0),
        on_tile=lambda *_: marks.append(time.time() - t0),
    )  # fmt: skip
    total = time.time() - t0
    mb, req = meter.bytes / 1e6, meter.requests
    print(f"{label:<44}{mb:6.1f} MB {req:4d} req  first result {marks[0]:5.1f}s  total {total:5.1f}s  "
          f"({len(res.mounds)} mounds, {len(res.pits)} pits, {len(res.lines)} lines)")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mbit", type=float, default=20.0, help="link speed in megabit/s")
    ap.add_argument("--latency-ms", type=float, default=60.0)
    ap.add_argument("--cog", default=DEFAULT_COG, help="a cloud-optimised GeoTIFF to serve and scan")
    a = ap.parse_args()
    COG = Path(a.cog)
    with rasterio.open(COG) as ds:
        bbox = tuple(ds.bounds)
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 1e6
    server, meter, base = serve(COG, a.mbit * 125_000, a.latency_ms / 1000)
    uid = (f"f{i}.tif" for i in itertools.count())
    cfg = ScanConfig(mound_fn=unet_mound_fn("output/unet.pt"), max_mounds_per_km2=120)
    size = COG.stat().st_size / 1e6
    print(f"link {a.mbit:g} Mbit/s, {a.latency_ms:g} ms latency; file {size:.1f} MB; {area:.2f} km2")
    meter.reset()
    t0 = time.time()
    urllib.request.urlopen(f"{base}/whole.tif").read()
    dl = time.time() - t0
    mb, req = meter.bytes / 1e6, meter.requests
    t0 = time.time()
    local = GeoTiffSource(str(COG), "CC BY 4.0", "x")
    plan = replace(make_plan(required_halo((1,), cfg.mound_scales), io_workers=1), tile_core=1024)
    scan(local, bbox, plan, cfg, log=lambda *_: None, gate=MemoryGate(0))
    compute = time.time() - t0
    label = "download whole file, THEN process"
    tot = dl + compute
    print(f"{label:<44}{mb:6.1f} MB {req:4d} req  first result {tot:5.1f}s  total {tot:5.1f}s  "
          f"(download {dl:.1f}s + compute {compute:.1f}s)")  # fmt: skip
    for w in (1, 4):
        run(f"stream tiles, {w} parallel read(s), 1 m", meter, base, uid, cfg, w, bbox)
    coarse = replace(cfg, cell=4.0, scales=(1, 2), mound_scales=(1, 2))
    run("overview-only scan, 4 m cells", meter, base, uid, coarse, 4, bbox, core=256, scales=(1, 2))
    server.shutdown()


if __name__ == "__main__":
    main()
