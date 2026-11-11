"""Geometric detectors for structure types a mound detector cannot see:
depressions (pits, ponds, pond barrows, dolines) and linear earthworks (banks, ditches, dykes).
Linear output includes modern roads and hedge banks: the terrain alone cannot tell them apart."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.feature import blob_log, hessian_matrix, hessian_matrix_eigvals
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from .features import fill_nodata


def _closed(z, r, c, rad, depth, n=16, frac=0.9):
    """A pit is higher than its centre in (nearly) every direction; a valley side is not."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = np.clip((r + 2.0 * rad * np.sin(a)).astype(int), 0, z.shape[0] - 1)
    cc = np.clip((c + 2.0 * rad * np.cos(a)).astype(int), 0, z.shape[1] - 1)
    return np.mean(z[rr, cc] - z[int(r), int(c)] >= 0.5 * depth) >= frac


@dataclass
class Depression:
    row: float
    col: float
    radius_m: float
    depth_m: float


def depressions(dtm, min_depth=0.2, sigmas=(3.0, 5.0, 8.0, 12.0), threshold=0.12):
    """Closed pits. `threshold` is the blob response in metres of depression, absolute, so a tile gives the
    same answer inside a mosaic as on its own."""
    z = fill_nodata(dtm).astype("float64")
    relief = ndimage.gaussian_filter(z, 2.0) - ndimage.gaussian_filter(z, 30.0)
    inv = np.clip(-relief, 0, None)
    out = []
    for r, c, s in blob_log(
        inv, min_sigma=sigmas[0], max_sigma=sigmas[-1], num_sigma=len(sigmas), threshold=threshold
    ):
        depth = float(-relief[int(r), int(c)])
        rad = float(s * 2**0.5)
        if depth >= min_depth and not np.isnan(dtm[int(r), int(c)]) and _closed(z, r, c, rad, depth):
            out.append(Depression(float(r), float(c), rad, depth))
    return sorted(out, key=lambda d: -d.depth_m)


@dataclass
class Line:
    kind: str  # "bank" | "ditch"
    length_m: float  # bounding-box diagonal, in pixels at 1 m cells
    pixels: np.ndarray  # (n, 2) row, col

    @property
    def extent(self) -> float:
        return float(np.hypot(*(self.pixels.max(0) - self.pixels.min(0) + 1)))


# banks / ditches: (low, high) hysteresis levels in metres of curvature x sigma^2, the median of the
# 88th / 97th percentiles over the five benchmark tiles. Absolute, so tiles agree at seams.
DEFAULT_THRESHOLDS = {"bank": (0.06, 0.13), "ditch": (0.066, 0.14)}


def line_fragments(dtm, sigma=3.0, thresholds=None, border=15, min_pixels=5):
    """Skeletons of ridge (bank) and valley (ditch) responses, unfiltered. `thresholds` maps kind to
    (low, high); None uses per-tile percentiles (88 / 97), which adapt to the tile but are not tile-safe."""
    z = fill_nodata(dtm).astype("float64")
    H = hessian_matrix(z, sigma=sigma, order="rc", use_gaussian_derivatives=True)
    l1, l2 = hessian_matrix_eigvals(H)
    # curvature across the line must dominate curvature along it, otherwise it is a blob
    bank = np.clip(-l2, 0, None) * sigma**2 * (np.abs(l1) < 0.5 * np.abs(l2))
    ditch = np.clip(l1, 0, None) * sigma**2 * (np.abs(l2) < 0.5 * np.abs(l1))
    ok = ~ndimage.binary_dilation(np.isnan(dtm), iterations=border)
    ok[:border], ok[-border:], ok[:, :border], ok[:, -border:] = False, False, False, False
    out: list[Line] = []
    for kind, strength in (("bank", bank), ("ditch", ditch)):
        strength = np.where(ok, strength, 0)
        if thresholds:
            lo, hi = thresholds[kind]
        else:
            pos = strength[strength > 0]
            lo, hi = np.percentile(pos, 88), np.percentile(pos, 97)
        mask = apply_hysteresis_threshold(strength, lo, hi)
        lab = label(skeletonize(mask), connectivity=2)
        out += [Line(kind, 0.0, r.coords) for r in regionprops(lab) if r.area >= min_pixels]
    return out


def keep_line(line: Line, min_extent: float = 80, straightness: float = 0.5) -> bool:
    """Long and straight: rejects rings, tangles and short noise."""
    return line.extent >= min_extent and line.extent / len(line.pixels) >= straightness


def linear_features(dtm, sigma=3.0, min_extent=80, straightness=0.5, border=15, thresholds=None):
    lines = [
        Line(f.kind, f.extent, f.pixels)
        for f in line_fragments(dtm, sigma, thresholds, border)
        if keep_line(f, min_extent, straightness)
    ]
    return sorted(lines, key=lambda l: -l.length_m)
