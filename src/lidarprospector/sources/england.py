"""England: Environment Agency 1 m LiDAR composite DTM via WCS (Open Government Licence v3)."""

import numpy as np

from ..wcs import fetch_dtm
from .base import Dem


class EnglandEA:
    name = "England EA LiDAR composite DTM 1m"
    licence = "Open Government Licence v3.0"
    attribution = "(c) Environment Agency copyright and/or database right"
    epsg = 27700
    native_cell = 1.0

    def covers(self, lon: float, lat: float) -> bool:
        return -6.5 <= lon <= 2.0 and 49.8 <= lat <= 55.9  # rough: includes bits of Wales and Scotland

    def fetch(self, bbox, cell: float | None = None):
        if cell not in (None, 1.0):
            raise ValueError(f"{self.name} is gridded at 1 m; cell={cell} is not supported")
        xmin, ymin, xmax, ymax = (int(round(v)) for v in bbox)
        z, x0, y1 = fetch_dtm(xmin, ymin, xmax, ymax)
        return Dem(
            z.astype(np.float32), x0, y1, 1.0, self.epsg, self.name, self.licence, self.attribution, 1.0
        )
