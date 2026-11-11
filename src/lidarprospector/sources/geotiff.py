"""Any elevation GeoTIFF or cloud-optimised GeoTIFF, local or over HTTP (windowed reads, on-the-fly
reprojection). Covers every country that publishes a DEM as a COG: pass the file or URL, the licence
and attribution that the publisher requires, and a metric working CRS (default: UTM for the area)."""

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from .base import Dem, require_metric, utm_epsg


class GeoTiffSource:
    def __init__(self, path: str, licence: str, attribution: str, epsg: int | None = None, name: str = ""):
        self.path = path if "://" not in path else f"/vsicurl/{path}"
        self.licence, self.attribution = licence, attribution
        with rasterio.open(self.path) as ds:
            self._bounds, self._crs, self.native_cell = ds.bounds, ds.crs, float(abs(ds.res[0]))
            self._nodata = ds.nodata
            to_ll = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            self._ll = to_ll.transform_bounds(*ds.bounds)
            self._geographic = ds.crs.is_geographic
            lon, lat = (self._ll[0] + self._ll[2]) / 2, (self._ll[1] + self._ll[3]) / 2
        if epsg is None:
            epsg = utm_epsg(lon, lat) if self._geographic else self._crs.to_epsg()
        self.epsg = require_metric(epsg)
        if self._geographic:  # native_cell is in degrees: express in metres at the centre latitude
            self.native_cell *= 111_320.0 * np.cos(np.radians(lat))
        self.name = name or f"GeoTIFF {path.rsplit('/', 1)[-1]}"

    def covers(self, lon: float, lat: float) -> bool:
        return self._ll[0] <= lon <= self._ll[2] and self._ll[1] <= lat <= self._ll[3]

    def fetch(self, bbox, cell: float | None = None) -> Dem:
        cell = cell or self.native_cell
        xmin, ymin, xmax, ymax = bbox
        w, h = round((xmax - xmin) / cell), round((ymax - ymin) / cell)
        dst = from_origin(xmin, ymax, cell, cell)
        with rasterio.open(self.path) as ds, WarpedVRT(
            ds, crs=f"EPSG:{self.epsg}", transform=dst, width=w, height=h,
            resampling=Resampling.bilinear, nodata=np.nan, dtype="float32",
        ) as vrt:  # fmt: skip
            z = vrt.read(1)
        if self._nodata is not None:
            z[z == self._nodata] = np.nan
        return Dem(
            z, xmin, ymax, cell, self.epsg, self.name, self.licence, self.attribution, self.native_cell
        )
