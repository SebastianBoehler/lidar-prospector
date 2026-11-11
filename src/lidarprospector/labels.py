"""Historic England scheduled monuments (OGL) as ground-truth labels, in EPSG:27700."""

import json
import re

import requests

URL = (
    "https://services-eu1.arcgis.com/ZOdPfBS3aqqDYPUQ/arcgis/rest/services/"
    "National_Heritage_List_for_England_NHLE_v02_VIEW/FeatureServer/6/query"
)
BARROW = re.compile(r"barrow|tumul|cairn", re.I)  # cairns are the upland equivalent of barrows
LINEAR = re.compile(
    r"cursus|avenue|dyke|linear boundar|boundary (bank|ditch)|trackway|ridgeway|lynchet", re.I
)
FIELDS = re.compile(r"field system|cultivation|celtic field", re.I)
ENCLOSURE = re.compile(
    r"hillfort|hill fort|enclosure|camp\b|henge|hengi|ringwork|settlement|castle|motte|\bfort\b", re.I
)


def classify(name: str | None, area_ha: float | None) -> str:
    """Coarse class from the monument name. Names often list several things ("The Cursus, two round
    barrows ..."), so a large protected area is classed by its structural keywords before 'barrow';
    a small one is a barrow if the name says so."""
    n = name or ""
    if area_ha is not None and area_ha <= 0.5 and BARROW.search(n):
        return "barrow"
    for label, rx in (("linear", LINEAR), ("fields", FIELDS), ("enclosure", ENCLOSURE), ("barrow", BARROW)):
        if rx.search(n):
            return label
    return "other"


def monuments(xmin, ymin, xmax, ymax):
    """All scheduled monuments intersecting the window, with a coarse class from the name."""
    r = requests.get(
        URL,
        params={
            "where": "1=1",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "27700",
            "outSR": "27700",
            "geometry": json.dumps(
                {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "spatialReference": {"wkid": 27700}}
            ),
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "ListEntry,Name,area_ha",
            "returnGeometry": "true",
            "f": "json",
        },
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(body["error"])
    return [
        {
            "id": f["attributes"]["ListEntry"],
            "name": f["attributes"]["Name"],
            "area_ha": f["attributes"]["area_ha"],
            "cls": classify(f["attributes"]["Name"], f["attributes"]["area_ha"]),
            "rings": f["geometry"]["rings"],
        }
        for f in body["features"]
    ]


def is_tight_barrow(m: dict, max_area_ha: float = 0.5) -> bool:
    """A single-barrow polygon: small protected area, so the polygon is the mound and not a whole
    cemetery or enclosure. Records without a recorded area are never tight."""
    return m["cls"] == "barrow" and m["area_ha"] is not None and m["area_ha"] <= max_area_ha


def barrows(xmin, ymin, xmax, ymax, max_area_ha: float = 0.5):
    return [m for m in monuments(xmin, ymin, xmax, ymax) if is_tight_barrow(m, max_area_ha)]
