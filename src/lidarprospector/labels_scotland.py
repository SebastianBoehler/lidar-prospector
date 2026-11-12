"""Canmore (Historic Environment Scotland's national record, Open Government Licence) as labels and as the
"is it already recorded?" check. Points, not polygons, with a stated positional accuracy.

`data/scotland/canmore/Canmore_Points.dbf` is read once and cached as a compact table. Each site gets one
of the benchmark types from its SITETYPE text, or none. "Possible" records and clearance / marker cairns
are kept out of the typed set (they stay in the table, so a candidate near *any* record counts as already
recorded).
"""

import re
from pathlib import Path

import numpy as np

CANMORE_DBF = Path("data/scotland/canmore/Canmore_Points.dbf")
CACHE = Path("data/scotland/canmore_sites.npz")
TYPES = ("round barrow", "cairn", "hut circle", "stone setting")

ACCURACY = {"1m": 1, "10m": 10, "100m": 100, "1km": 1000}


def accuracy_m(text: str | None) -> int:
    m = re.search(r"nearest (\d+(?:m|km))", text or "")
    return ACCURACY.get(m.group(1), 5000) if m else 5000


def site_type(sitetype: str | None) -> str | None:
    """Benchmark type of a Canmore SITETYPE string, or None. Several comma-separated terms may be present."""
    s = (sitetype or "").upper()
    if "POSSIBLE" in s:
        return None
    if re.search(r"HUT CIRCLE", s):
        return "hut circle"
    if re.search(r"STANDING STONE|STONE CIRCLE|STONE ROW|STONE SETTING", s):
        return "stone setting"
    if re.search(r"\bBARROW", s) and "LONG BARROW" not in s:
        return "round barrow"
    if re.search(r"\bCAIRN\b", s) and not re.search(r"CLEARANCE|MARKER|CHAMBERED|CAIRNFIELD|SHOOTING", s):
        return "cairn"
    return None


def build(dbf: Path = CANMORE_DBF, cache: Path = CACHE) -> dict[str, np.ndarray]:
    """Read the dbf once (a few seconds) and cache the fields that matter."""
    import shapefile  # pyshp

    with dbf.open("rb") as fh:
        reader = shapefile.Reader(dbf=fh)
        names = [f[0] for f in reader.fields[1:]]
        want = ["CANMOREID", "NMRSNAME", "SITETYPE", "ACCURACY", "ARCHAEOLOG", "XCOORD", "YCOORD"]
        idx = [names.index(k) for k in want]
        rows = [tuple(rec[i] for i in idx) for rec in reader.iterRecords()]
    table = {
        "id": np.array([r[0] for r in rows], np.int64),
        "name": np.array([r[1] or "" for r in rows]),
        "sitetype": np.array([r[2] or "" for r in rows]),
        "accuracy": np.array([accuracy_m(r[3]) for r in rows], np.int32),
        "archaeology": np.array([r[4] == "Y" for r in rows]),
        "x": np.array([r[5] for r in rows], np.float64),
        "y": np.array([r[6] for r in rows], np.float64),
    }
    table["type"] = np.array([site_type(s) or "" for s in table["sitetype"]])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **table)
    return table


def load(cache: Path = CACHE) -> dict[str, np.ndarray]:
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    return build()


def in_bbox(table: dict[str, np.ndarray], bbox) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox
    return (table["x"] >= xmin) & (table["x"] < xmax) & (table["y"] >= ymin) & (table["y"] < ymax)
