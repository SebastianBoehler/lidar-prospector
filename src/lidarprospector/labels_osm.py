"""OpenStreetMap tumulus nodes as point labels (ODbL). Crowd-sourced and unscheduled: positions are
good to a few metres where mappers used lidar, worse elsewhere, and coverage is incomplete."""

import numpy as np
import requests
from pyproj import Transformer
from requests.adapters import HTTPAdapter
from scipy import ndimage
from urllib3.util.retry import Retry

OVERPASS = "https://overpass-api.de/api/interpreter"


def tumuli(bbox, epsg: int) -> np.ndarray:
    """(n, 2) CRS x, y of tumulus nodes inside bbox (CRS metres)."""
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    to_crs = Transformer.from_crs(4326, epsg, always_xy=True)
    lon0, lat0, lon1, lat1 = to_ll.transform_bounds(*bbox)
    b = f"({lat0},{lon0},{lat1},{lon1})"
    q = (
        "[out:json][timeout:90];("
        + "".join(f'node["{k}"="tumulus"]{b};' for k in ("historic", "archaeological_site", "site_type"))
        + ");out;"
    )
    s = requests.Session()
    retry = Retry(total=6, backoff_factor=4, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=None)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    r = s.post(OVERPASS, data={"data": q}, headers={"User-Agent": "lidar-prospector/0.0"}, timeout=180)
    r.raise_for_status()
    pts = {e["id"]: to_crs.transform(e["lon"], e["lat"]) for e in r.json()["elements"]}
    xy = np.array(list(pts.values())).reshape(-1, 2)
    inside = (xy[:, 0] >= bbox[0]) & (xy[:, 0] < bbox[2]) & (xy[:, 1] >= bbox[1]) & (xy[:, 1] < bbox[3])
    return xy[inside]


def point_labels(shape, x0: float, y1: float, cell: float, xy: np.ndarray, tol_m: float = 15.0):
    """`near` raster: pixel value = 1-based id of the nearest point within tol_m, else 0 (same convention
    as the polygon labels, so the benchmarks can score both)."""
    ids = np.zeros(shape, np.int32)
    for i, (x, y) in enumerate(xy, 1):
        r, c = int((y1 - y) / cell), int((x - x0) / cell)
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            ids[r, c] = i
    dist, (ri, ci) = ndimage.distance_transform_edt(ids == 0, return_indices=True)
    return np.where(dist * cell <= tol_m, ids[ri, ci], 0)
