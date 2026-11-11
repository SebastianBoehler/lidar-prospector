"""Datasets whose access, licence and citation were read from the publisher's own page. A preset only
saves typing: it builds a GeoTiffSource with the terms attached, and says what has and has not been tried."""

from .geotiff import GeoTiffSource

PRESETS = {
    "quirigua": {
        "url": "https://opentopography.s3.sdsc.edu/dataspace/OTDS.072026.32616.7/raster/quirigua_museo_dtm_cog.tif",
        "licence": "CC BY 4.0",
        "attribution": (
            "Maurer, J. (2026). UAV Lidar Survey of Museo Quirigua, Guatemala, January 2024. Distributed by "
            "OpenTopography. https://doi.org/10.5069/G9416V95. Data collected under an NSF planning grant "
            "(Obrist-Farner & Maurer, Missouri S&T)."
        ),
        "name": "Quirigua Museo UAV lidar DTM 0.25 m (Guatemala, under canopy, 0.71 km2)",
        "note": "published page read; the file itself (27 MB) has not been fetched by this project yet",
    },
}


def preset_source(key: str) -> GeoTiffSource:
    if key not in PRESETS:
        raise SystemExit(f"unknown preset {key!r}; available: {', '.join(PRESETS)}")
    p = PRESETS[key]
    return GeoTiffSource(p["url"], p["licence"], p["attribution"], name=p["name"])
