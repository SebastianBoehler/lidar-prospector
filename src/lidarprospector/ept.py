"""Minimal Entwine Point Tile (EPT) client: fetch only the octree nodes that
intersect a window, so a few km^2 come out of a multi-billion-point project."""

from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import laspy
import numpy as np
import requests
from pyproj import Transformer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://usgs-lidar-public.s3.amazonaws.com"
WEB_MERCATOR = 3857


@dataclass(frozen=True)
class Node:
    key: str
    points: int

    @property
    def depth(self) -> int:
        return int(self.key.split("-")[0])


def window_3857(lat: float, lon: float, half_m: float) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) in EPSG:3857 for a square of 2*half_m ground metres."""
    x, y = Transformer.from_crs(4326, WEB_MERCATOR, always_xy=True).transform(lon, lat)
    h = half_m / math.cos(math.radians(lat))
    return x - h, y - h, x + h, y + h


def _node_box(root: list[float], key: str) -> tuple[float, float, float, float]:
    d, i, j, _ = (int(v) for v in key.split("-"))
    size = (root[3] - root[0]) / 2**d
    return root[0] + i * size, root[1] + j * size, root[0] + (i + 1) * size, root[1] + (j + 1) * size


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


class EptProject:
    def __init__(self, name: str):
        self.name = name
        self.url = f"{BASE}/{name}"
        self.session = requests.Session()
        retry = Retry(total=6, backoff_factor=1.0, status_forcelist=(500, 502, 503, 504))
        self.session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
        self.meta = self._json(f"{self.url}/ept.json")
        self.root = self.meta["bounds"]
        if self.meta["srs"].get("horizontal") != str(WEB_MERCATOR):
            raise ValueError(f"{name}: expected EPSG:3857 storage, got {self.meta['srs']}")

    def _json(self, url: str) -> dict:
        r = self.session.get(url, timeout=60)
        r.raise_for_status()
        return r.json()

    def nodes(self, win) -> list[Node]:
        """All octree nodes intersecting `win`, walking the hierarchy pages lazily."""
        out: list[Node] = []
        pages = ["0-0-0-0"]
        while pages:
            page = self._json(f"{self.url}/ept-hierarchy/{pages.pop()}.json")
            for key, count in page.items():
                if not _overlaps(_node_box(self.root, key), win):
                    continue
                if count == -1:
                    pages.append(key)
                elif count > 0:
                    out.append(Node(key, count))
        return out

    def sizes(self, nodes: list[Node]) -> int:
        def head(n: Node) -> int:
            r = self.session.head(f"{self.url}/ept-data/{n.key}.laz", timeout=60)
            r.raise_for_status()
            return int(r.headers["Content-Length"])

        with ThreadPoolExecutor(16) as ex:
            return sum(ex.map(head, nodes))

    def fetch(self, nodes: list[Node], win, utm_epsg: int):
        """Download nodes, clip to `win`, return x, y, z (UTM metres) and classification."""
        to_utm = Transformer.from_crs(WEB_MERCATOR, utm_epsg, always_xy=True)

        def get(n: Node):
            r = self.session.get(f"{self.url}/ept-data/{n.key}.laz", timeout=120)
            r.raise_for_status()
            las = laspy.read(io.BytesIO(r.content))
            x, y = np.asarray(las.x), np.asarray(las.y)
            keep = (x >= win[0]) & (x <= win[2]) & (y >= win[1]) & (y <= win[3])
            return x[keep], y[keep], np.asarray(las.z)[keep], np.asarray(las.classification)[keep]

        with ThreadPoolExecutor(8) as ex:
            parts = list(ex.map(get, nodes))
        x, y, z, c = (np.concatenate(p) for p in zip(*parts, strict=True))
        ux, uy = to_utm.transform(x, y)
        return ux, uy, z, c
