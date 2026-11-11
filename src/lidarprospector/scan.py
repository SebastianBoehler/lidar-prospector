"""One command: pick a terrain source, plan for this machine, scan an area, write GeoJSON.

uv run python -m lidarprospector.scan --source england --lonlat -1.8262 51.1789 --size-km 6
uv run python -m lidarprospector.scan --source geotiff:/path/dem.tif --licence "..." --attribution "..."
"""

import argparse
import json
from pathlib import Path

from pyproj import Transformer

from .pipeline import ScanConfig, scan
from .runtime import make_plan
from .sources import EnglandEA, GeoTiffSource, UsgsEpt, source_for
from .tiling import required_halo, scales_for


def build_source(args):
    spec = args.source
    if spec == "england":
        return EnglandEA()
    if spec.startswith("ept:"):
        return UsgsEpt(spec[4:])
    if spec.startswith("geotiff:"):
        if not (args.licence and args.attribution):
            raise SystemExit(
                "--licence and --attribution are required for a GeoTIFF source: use the publisher's terms"
            )
        return GeoTiffSource(spec[8:], args.licence, args.attribution, epsg=args.epsg)
    if spec == "auto":
        if not args.lonlat:
            raise SystemExit("--source auto needs --lonlat")
        return source_for(*args.lonlat)
    raise SystemExit(f"unknown source {spec!r}: use england, ept:PROJECT, geotiff:PATH_OR_URL or auto")


def build_bbox(args, source):
    if args.bbox:
        return tuple(args.bbox)
    if not args.lonlat:
        raise SystemExit("give --bbox (source CRS metres) or --lonlat with --size-km")
    x, y = Transformer.from_crs(4326, source.epsg, always_xy=True).transform(*args.lonlat)
    half = args.size_km * 500.0
    return (round(x - half), round(y - half), round(x + half), round(y + half))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="auto", help="england | ept:PROJECT | geotiff:PATH_OR_URL | auto")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    ap.add_argument("--lonlat", type=float, nargs=2, metavar=("LON", "LAT"))
    ap.add_argument("--size-km", type=float, default=3.0)
    ap.add_argument("--licence", help="licence text for a GeoTIFF source")
    ap.add_argument("--attribution", help="attribution text for a GeoTIFF source")
    ap.add_argument("--epsg", type=int, help="metric working CRS for a GeoTIFF source (default: UTM)")
    ap.add_argument("--min-size-m", type=float, default=5.0, help="narrowest structure to look for")
    ap.add_argument(
        "--max-size-m", type=float, default=30.0, help="widest structure; larger adds coarser pyramid levels"
    )
    ap.add_argument("--mounds", type=Path, help="U-Net weights from train_unet; omit to skip mound detection")
    ap.add_argument(
        "--mounds-per-km2", type=float, default=22.0, help="mound candidates kept per km2, best first"
    )
    ap.add_argument("--mask-roads", action="store_true", help="drop lines along OpenStreetMap roads (ODbL)")
    ap.add_argument("--ram-fraction", type=float, default=0.5, help="share of currently free RAM to use")
    ap.add_argument("--max-budget-gb", type=float, help="hard ceiling on memory use")
    ap.add_argument("--io-workers", type=int, default=4, help="parallel downloads")
    ap.add_argument("--out", type=Path, default=Path("output/scan.geojson"))
    args = ap.parse_args()

    source = build_source(args)
    bbox = build_bbox(args, source)
    scales = scales_for(args.min_size_m, args.max_size_m)
    plan = make_plan(
        required_halo(scales), args.ram_fraction, max_budget_gb=args.max_budget_gb, io_workers=args.io_workers
    )
    print(plan.describe())
    print(f"pyramid       scales {scales} for structures {args.min_size_m:g}-{args.max_size_m:g} m")
    print(f"area          {bbox} (EPSG:{source.epsg}); data: {source.licence}; {source.attribution}\n")
    cfg = ScanConfig(scales=scales, max_mounds_per_km2=args.mounds_per_km2)
    if args.mask_roads:
        from .roads import RoadIndex, road_polylines

        cfg.roads = RoadIndex(road_polylines(bbox, source.epsg))
    if args.mounds:
        from .mounds import unet_mound_fn

        cfg.mound_fn = unet_mound_fn(args.mounds, plan.hardware.device)
    res = scan(source, bbox, plan, cfg)
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(res.to_geojson()))
    print(f"\n{len(res.mounds)} mounds, {len(res.pits)} pits, {len(res.lines)} lines -> {args.out}")
    print(f"waited {res.seconds_waiting_for_data:.0f}s for data, computed {res.seconds_computing:.0f}s")


if __name__ == "__main__":
    main()
