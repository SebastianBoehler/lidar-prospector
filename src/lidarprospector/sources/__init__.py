"""Terrain sources and a registry that picks one for a location."""

from .base import Dem, Source, utm_epsg
from .england import EnglandEA
from .ept import UsgsEpt
from .geotiff import GeoTiffSource

REGISTRY: list[Source] = [EnglandEA()]


def source_for(lon: float, lat: float, extra: list[Source] | None = None) -> Source:
    for src in [*(extra or []), *REGISTRY]:
        if src.covers(lon, lat):
            return src
    raise LookupError(
        f"no registered terrain source covers lon={lon:.3f}, lat={lat:.3f}. Registered: "
        f"{[s.name for s in REGISTRY]}. Pass a DEM yourself with --geotiff PATH_OR_URL --licence TEXT."
    )


__all__ = ["REGISTRY", "Dem", "EnglandEA", "GeoTiffSource", "Source", "UsgsEpt", "source_for", "utm_epsg"]
