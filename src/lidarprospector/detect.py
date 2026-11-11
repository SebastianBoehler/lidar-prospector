"""Candidate earthwork detector: positive local relief, split into compact mounds and
linear ridges, with natural escarpments and long straight roads suppressed."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops

from .terrain import local_relief


@dataclass
class Candidate:
    id: int
    row: float
    col: float
    area_m2: float
    peak_m: float
    mean_m: float
    elongation: float
    solidity: float
    kind: str  # "mound" | "ridge"
    score: float


def steep_mask(dtm, cell=1.0, slope_deg=8.0, buffer_m=25.0):
    """Erosional terrain (bluffs, gullies): steep ground plus a buffer around it. Earthworks
    sit on gentle ground, so local relief inside this zone is not evidence of construction."""
    filled = np.where(np.isnan(dtm), np.nanmean(dtm), dtm)
    gy, gx = np.gradient(ndimage.gaussian_filter(filled, 2.0 / cell), cell)
    steep = np.degrees(np.arctan(np.hypot(gx, gy))) > slope_deg
    return ndimage.binary_dilation(steep, iterations=int(buffer_m / cell))


def _components(dtm, cell, min_height, min_area, max_area, suppress_erosion):
    lrm = local_relief(dtm, sigma_m=15.0, cell=cell)
    anomalous = lrm > min_height
    if suppress_erosion:
        anomalous &= ~steep_mask(dtm, cell)
    anomalous = ndimage.binary_opening(anomalous, iterations=1)
    labels = label(anomalous, connectivity=2)
    out: list[Candidate] = []
    for r in regionprops(labels, intensity_image=np.nan_to_num(lrm)):
        area = r.area * cell**2
        if not (min_area <= area <= max_area):
            continue
        minor = max(r.axis_minor_length, 1e-6)
        elong = r.axis_major_length / minor
        kind = "ridge" if elong > 4 else "mound"
        # compact, well-defined, taller anomalies rank higher
        score = float(r.intensity_max * min(area, 3000) ** 0.5 * (0.5 + r.solidity))
        out.append(
            Candidate(
                len(out),
                r.centroid[0],
                r.centroid[1],
                area,
                float(r.intensity_max),
                float(r.intensity_mean),
                float(elong),
                float(r.solidity),
                kind,
                score,
            )
        )
    return out, lrm, labels


def detect(dtm, cell=1.0):
    """Low relief on gentle ground: ridges, platforms, small mounds. Known miss: large
    mounds with steep flanks (e.g. Poverty Point Mound A) are discarded with the erosion mask."""
    out, lrm, labels = _components(dtm, cell, 0.15, 60.0, 250000.0, True)
    out.sort(key=lambda c: -c.score)
    for i, c in enumerate(out):
        c.id = i
    return out, lrm, labels
