"""Netherlands: AHN DTM 0.5 m from PDOK (WCS 2.0 with the scaling extension), CC0, no login.

The service can resample on the server, so a 1 m working grid costs a quarter of the native download."""

from pathlib import Path

import numpy as np
import rasterio
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import Dem

URL = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
COVERAGE = "dtm_05m"
NATIVE = 0.5


def request_params(bbox, cell: float) -> dict:
    """GetCoverage KVP parameters for a window on a `cell`-metre grid (server-side scaling)."""
    xmin, ymin, xmax, ymax = bbox
    if cell < NATIVE:
        raise ValueError(f"AHN is 0.5 m; cannot serve a {cell} m grid")
    return {
        "service": "WCS", "version": "2.0.1", "request": "GetCoverage", "coverageId": COVERAGE,
        "format": "image/tiff", "subset": [f"x({xmin},{xmax})", f"y({ymin},{ymax})"],
        "scalefactor": f"{NATIVE / cell:g}",
    }  # fmt: skip


class NetherlandsAHN:
    name = "Netherlands AHN DTM (Rijkswaterstaat)"
    licence = "CC0 1.0"
    attribution = "Actueel Hoogtebestand Nederland (AHN), Rijkswaterstaat, via PDOK"
    epsg = 28992
    native_cell = NATIVE

    def __init__(self, cache: Path = Path("data")):
        self.cache = cache

    def covers(self, lon: float, lat: float) -> bool:
        return 3.2 <= lon <= 7.3 and 50.7 <= lat <= 53.6

    def fetch(self, bbox, cell: float | None = None) -> Dem:
        cell = cell or 1.0
        bbox = tuple(int(round(v)) for v in bbox)
        self.cache.mkdir(exist_ok=True)
        path = self.cache / f"ahn_dtm_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{cell:g}m.tif"
        if not path.exists():
            s = requests.Session()
            retry = Retry(total=5, backoff_factor=5, status_forcelist=(502, 503, 504))
            s.mount("https://", HTTPAdapter(max_retries=retry))
            r = s.get(URL, params=request_params(bbox, cell), timeout=600)
            r.raise_for_status()
            if r.headers.get("content-type", "").startswith(("application/xml", "text/xml")):
                raise RuntimeError(r.text[:500])
            path.write_bytes(r.content)
        with rasterio.open(path) as ds:
            z = ds.read(1).astype("float32")
            if ds.nodata is not None:
                z[z == ds.nodata] = np.nan
            z[np.abs(z) > 1e6] = np.nan  # server fill values
            t = ds.transform
            if abs(abs(t.a) - cell) > 1e-6 or abs(abs(t.e) - cell) > 1e-6:
                raise ValueError(f"expected {cell} m cells, got {t}")
            return Dem(z, t.c, t.f, cell, self.epsg, self.name, self.licence, self.attribution, NATIVE)
