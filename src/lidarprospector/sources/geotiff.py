"""Any elevation GeoTIFF or cloud-optimised GeoTIFF, local or over HTTP (windowed reads, on-the-fly
reprojection). Covers every country that publishes a DEM as a COG: pass the file or URL, the licence
and attribution that the publisher requires, and a metric working CRS (default: UTM for the area)."""

import os
import threading

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

from .base import Dem, require_metric, utm_epsg

# GDAL's HTTP driver lists a file's parent directory when opening it, one wasted (and on some servers very
# large) request per open, from every worker thread. Process-wide defaults, so threads inherit them.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", str(64 << 20))
# Decoded-block cache (MB). One dataset handle stays open per source, so neighbouring tiles reuse blocks they
# share instead of downloading the halo overlap again (measured without it: 2.3x the file).
os.environ.setdefault("GDAL_CACHEMAX", "256")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")


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
        self._ds = None
        self._lock = threading.Lock()
        self.name = name or f"GeoTIFF {path.rsplit('/', 1)[-1]}"

    def covers(self, lon: float, lat: float) -> bool:
        return self._ll[0] <= lon <= self._ll[2] and self._ll[1] <= lat <= self._ll[3]

    def fetch(self, bbox, cell: float | None = None) -> Dem:
        cell = cell or self.native_cell
        xmin, ymin, xmax, ymax = bbox
        w, h = round((xmax - xmin) / cell), round((ymax - ymin) / cell)
        # reducing resolution must average the source cells; bilinear would alias fine detail
        resampling = Resampling.average if cell >= 1.5 * self.native_cell else Resampling.bilinear
        if not self._geographic and self._crs.to_epsg() == self.epsg:
            z = self._read_direct(bbox, (h, w), resampling)
        else:
            z = self._read_warped(xmin, ymax, cell, (h, w), resampling)
        if self._nodata is not None:
            z[z == self._nodata] = np.nan
        meta = (self.name, self.licence, self.attribution)
        return Dem(z, xmin, ymax, cell, self.epsg, *meta, self.native_cell)

    def _read_direct(self, bbox, shape, resampling) -> np.ndarray:
        """Same CRS: read the window straight from the file. A decimated read lets GDAL pick the matching
        overview level, so a coarse look at a cloud GeoTIFF costs a few percent of the bytes (a warped
        read would average from full resolution and fetch everything)."""
        xmin, ymin, xmax, ymax = bbox
        out = np.full(shape, np.nan, np.float32)
        with self._lock:  # one shared handle: its block cache is what avoids re-downloading overlaps
            if self._ds is None:
                self._ds = rasterio.open(self.path)
            ds = self._ds
            b = ds.bounds
            ix0, iy0, ix1, iy1 = max(xmin, b.left), max(ymin, b.bottom), min(xmax, b.right), min(ymax, b.top)
            if ix1 <= ix0 or iy1 <= iy0:
                return out
            win = from_bounds(ix0, iy0, ix1, iy1, ds.transform)
            # output pixels covered by the intersection, aligned to the requested grid
            sx, sy = shape[1] / (xmax - xmin), shape[0] / (ymax - ymin)
            c0, c1 = round((ix0 - xmin) * sx), round((ix1 - xmin) * sx)
            r0, r1 = round((ymax - iy1) * sy), round((ymax - iy0) * sy)
            if c1 > c0 and r1 > r0:
                block = ds.read(1, window=win, out_shape=(r1 - r0, c1 - c0), resampling=resampling)
                out[r0:r1, c0:c1] = block
        return out

    def close(self) -> None:
        with self._lock:
            if self._ds is not None:
                self._ds.close()
                self._ds = None

    def _read_warped(self, xmin, ymax, cell, shape, resampling) -> np.ndarray:
        h, w = shape
        with (
            rasterio.open(self.path) as ds,
            WarpedVRT(
                ds,
                crs=f"EPSG:{self.epsg}",
                transform=from_origin(xmin, ymax, cell, cell),
                width=w,
                height=h,
                resampling=resampling,
                nodata=np.nan,
                dtype="float32",
            ) as vrt,
        ):
            return vrt.read(1)
