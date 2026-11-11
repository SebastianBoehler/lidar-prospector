"""Mapped roads and railways (OpenStreetMap, ODbL) rasterised onto the terrain grid, so modern
linear features are not reported as earthworks. Only the window's bounding box is sent."""

from pathlib import Path

import numpy as np
import requests
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from requests.adapters import HTTPAdapter
from scipy import ndimage
from urllib3.util.retry import Retry

OVERPASS = "https://overpass-api.de/api/interpreter"


def road_mask(shape, x0, y1, buffer_m=8, cache=Path("data")):
    cache.mkdir(exist_ok=True)
    path = cache / f"roads_{int(x0)}_{int(y1)}_{shape[0]}x{shape[1]}.npy"
    if not path.exists():
        np.save(path, _fetch_roads(shape, x0, y1))
    return ndimage.binary_dilation(np.load(path), iterations=buffer_m)


def _fetch_roads(shape, x0, y1):
    bng_to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    ll_to_bng = Transformer.from_crs(4326, 27700, always_xy=True)
    lon0, lat0 = bng_to_ll.transform(x0, y1 - shape[0])
    lon1, lat1 = bng_to_ll.transform(x0 + shape[1], y1)
    box = f"({lat0},{lon0},{lat1},{lon1})"
    q = f'[out:json][timeout:60];(way["highway"]{box};way["railway"]{box};);out geom;'
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=5, backoff_factor=3, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=None
            )
        ),
    )
    r = session.post(OVERPASS, data={"data": q}, headers={"User-Agent": "lidar-prospector/0.0"}, timeout=120)
    r.raise_for_status()
    lines = []
    for el in r.json()["elements"]:
        pts = [ll_to_bng.transform(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(pts) > 1:
            lines.append(({"type": "LineString", "coordinates": pts}, 1))
    if not lines:
        return np.zeros(shape, bool)
    return rasterize(lines, out_shape=shape, transform=from_origin(x0, y1, 1, 1), dtype="uint8") > 0
