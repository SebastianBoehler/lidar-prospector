"""Fetch a window of the Environment Agency 1 m LiDAR composite DTM (England, OGL) via WCS 2.0."""

from pathlib import Path

import numpy as np
import rasterio
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs"
COVERAGE = "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"


def fetch_dtm(xmin: int, ymin: int, xmax: int, ymax: int, cache: Path = Path("data")):
    """Return (dtm north-up float32 with NaN nodata, xmin, ymax) in EPSG:27700, 1 m cells."""
    cache.mkdir(exist_ok=True)
    path = cache / f"ea_dtm_{xmin}_{ymin}_{xmax}_{ymax}.tif"
    if not path.exists():
        session = requests.Session()
        session.mount(
            "https://",
            HTTPAdapter(max_retries=Retry(total=5, backoff_factor=5, status_forcelist=(502, 503, 504))),
        )
        r = session.get(
            URL,
            params={
                "service": "WCS",
                "version": "2.0.1",
                "request": "GetCoverage",
                "coverageId": COVERAGE,
                "subset": [f"E({xmin},{xmax})", f"N({ymin},{ymax})"],
                "format": "image/tiff",
            },
            timeout=600,
        )
        r.raise_for_status()
        if r.headers.get("content-type", "").startswith("application/xml"):
            raise RuntimeError(r.text[:500])
        path.write_bytes(r.content)
    with rasterio.open(path) as ds:
        dtm = ds.read(1).astype("float32")
        if ds.nodata is not None:
            dtm[dtm == ds.nodata] = np.nan
        t = ds.transform
        if abs(t.a - 1) > 1e-6 or abs(t.e + 1) > 1e-6:
            raise ValueError(f"expected 1 m north-up cells, got {t}")
        return dtm, t.c, t.f
