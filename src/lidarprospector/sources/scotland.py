"""Scottish Public Sector LiDAR: bare-earth DTM tiles in a public S3 bucket (Open Government Licence v3).

Phases 1-2 are 1 m tiles on the 10 km OS grid ('NH34'); phases 3-6 are 50 cm tiles on 5 km quarters
('NS16NE'). All are tiled LZW GeoTIFFs in EPSG:27700, so a window read fetches just the blocks it touches.
The bucket is listed once and the tile index cached under data/ (the portal's WCS is switched off, so the
files are read directly). Where a place is in several phases the later phase wins.
"""

import json
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests

from .base import Dem
from .geotiff import GeoTiffSource

BUCKET = "https://srsp-open-data.s3.eu-west-2.amazonaws.com/"
PHASES = tuple(f"phase-{n}" for n in range(1, 7))
TILE = re.compile(r"/([A-Z]{2}\d{2}(?:[NS][EW])?)_(?:1M|50CM)_DTM[^/]*\.tif$")
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # OS grid letters: no I
SIZE = 10_000
HALF = SIZE // 2  # phases 3-6 are 50 cm tiles in 5 km quarters named like NS16NE


def tile_origin(code: str) -> tuple[int, int]:
    """South-west corner (easting, northing) of a 10 km OS grid tile ('NH34') or a 5 km quarter ('NS16NE')."""
    l1, l2 = LETTERS.index(code[0]), LETTERS.index(code[1])
    e = ((l1 - 2) % 5) * 5 + l2 % 5
    n = 19 - (l1 // 5) * 5 - l2 // 5
    e, n = e * 100_000 + int(code[2]) * SIZE, n * 100_000 + int(code[3]) * SIZE
    if len(code) == 6:
        e, n = e + (HALF if code[5] == "E" else 0), n + (HALF if code[4] == "N" else 0)
    return e, n


def tile_code(e: int, n: int) -> str:
    """OS grid code ('NH34') of the 10 km tile containing easting / northing (metres)."""
    e100, n100 = e // 100_000, n // 100_000
    r = 19 - n100
    l1 = r - r % 5 + (e100 + 10) // 5
    l2 = r * 5 % 25 + e100 % 5
    return f"{LETTERS[l1]}{LETTERS[l2]}{e % 100_000 // SIZE}{n % 100_000 // SIZE}"


def quarter_code(e: int, n: int) -> str:
    """Code of the 5 km quarter containing the point, e.g. 'NS16NE'."""
    ns = "N" if n % SIZE >= HALF else "S"
    ew = "E" if e % SIZE >= HALF else "W"
    return f"{tile_code(e, n)}{ns}{ew}"


def list_phase(phase: str, session: requests.Session | None = None) -> dict[str, str]:
    """tile code -> https URL for one phase's DTM tiles (paged S3 listing, anonymous)."""
    s = session or requests.Session()
    prefix = f"lidar/{phase}/dtm/27700/gridded/"
    out, token = {}, None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = s.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {"s": root.tag.split("}")[0].strip("{")}
        for key in root.findall("s:Contents/s:Key", ns):
            m = TILE.search("/" + key.text)
            if m:
                out[m.group(1)] = BUCKET + key.text
        nxt = root.find("s:NextContinuationToken", ns)
        if nxt is None:
            return out
        token = nxt.text


def load_index(cache: Path = Path("data/srsp_index.json")) -> dict[str, str]:
    if cache.exists():
        return json.loads(cache.read_text())
    index: dict[str, str] = {}
    for phase in PHASES:
        index.update(list_phase(phase))
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(index))
    return index


class ScotlandSRSP:
    name = "Scottish Public Sector LiDAR (SRSP)"
    licence = "Open Government Licence v3.0"
    attribution = "Contains Scottish Government and JNCC data, Open Government Licence v3.0"
    epsg = 27700
    native_cell = 1.0

    def __init__(self, index: dict[str, str] | None = None):
        self._index = index
        self._sources: dict[str, GeoTiffSource] = {}
        self._lock = threading.Lock()

    @property
    def index(self) -> dict[str, str]:
        if self._index is None:
            self._index = load_index()
        return self._index

    def covers(self, lon: float, lat: float) -> bool:
        if not (-8.7 <= lon <= -0.7 and 54.6 <= lat <= 61.0):  # cheap test before any network listing
            return False
        from pyproj import Transformer

        x, y = Transformer.from_crs(4326, 27700, always_xy=True).transform(lon, lat)
        return quarter_code(int(x), int(y)) in self.index or tile_code(int(x), int(y)) in self.index

    def fetch(self, bbox, cell: float | None = None) -> Dem:
        cell = cell or 1.0
        xmin, ymin, xmax, ymax = bbox
        shape = (round((ymax - ymin) / cell), round((xmax - xmin) / cell))
        out = np.full(shape, np.nan, np.float32)
        for e in range(int(xmin) // HALF * HALF, int(xmax) // HALF * HALF + 1, HALF):
            for n in range(int(ymin) // HALF * HALF, int(ymax) // HALF * HALF + 1, HALF):
                url = self.index.get(quarter_code(e, n)) or self.index.get(tile_code(e, n))
                if url is None:
                    continue
                z = self._source(url).fetch(bbox, cell).z
                out = np.where(np.isnan(out), z, out)
        return Dem(out, xmin, ymax, cell, self.epsg, self.name, self.licence, self.attribution, 1.0)

    def _source(self, url: str) -> GeoTiffSource:
        with self._lock:
            if url not in self._sources:
                self._sources[url] = GeoTiffSource(url, self.licence, self.attribution, epsg=self.epsg)
            return self._sources[url]
