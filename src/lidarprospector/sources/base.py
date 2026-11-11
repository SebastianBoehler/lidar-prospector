"""Common shape for every terrain source: a metric-CRS, north-up elevation grid plus provenance."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pyproj import CRS, Transformer


@dataclass
class Dem:
    z: np.ndarray  # float32, north-up, NaN where there is no data
    x0: float  # CRS x of the grid's left edge
    y1: float  # CRS y of the grid's top edge
    cell: float  # metres per cell (square cells)
    epsg: int  # projected, metric CRS
    source: str
    licence: str
    attribution: str
    native_cell: float  # the source's own resolution, for quality warnings

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        h, w = self.z.shape
        return self.x0, self.y1 - h * self.cell, self.x0 + w * self.cell, self.y1

    def to_wgs84(self, row: float, col: float) -> tuple[float, float]:
        """(lon, lat) of a pixel position."""
        t = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        return t.transform(self.x0 + col * self.cell, self.y1 - row * self.cell)


class Source(Protocol):
    name: str
    licence: str
    attribution: str
    epsg: int  # working CRS (metric)
    native_cell: float

    def covers(self, lon: float, lat: float) -> bool: ...

    def fetch(self, bbox: tuple[float, float, float, float], cell: float | None = None) -> Dem:
        """Elevation for (xmin, ymin, xmax, ymax) in `epsg` metres, on a grid aligned to xmin / ymax.
        `cell` asks for a working resolution; sources that cannot resample raise if it differs from 1 m."""
        ...


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def require_metric(epsg: int) -> int:
    crs = CRS.from_epsg(epsg)
    if not crs.is_projected or crs.axis_info[0].unit_name not in ("metre", "meter"):
        raise ValueError(f"EPSG:{epsg} is not a projected metric CRS")
    return epsg
