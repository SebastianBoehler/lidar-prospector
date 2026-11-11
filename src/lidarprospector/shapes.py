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


def depressions(dtm, min_depth=0.2, sigmas=(3.0, 5.0, 8.0, 12.0)):
    z = fill_nodata(dtm).astype("float64")
    relief = ndimage.gaussian_filter(z, 2.0) - ndimage.gaussian_filter(z, 30.0)
    inv = np.clip(-relief, 0, None)
    out = []
    for r, c, s in blob_log(
        inv / max(inv.max(), 1e-6),
        min_sigma=sigmas[0],
        max_sigma=sigmas[-1],
        num_sigma=len(sigmas),
        threshold=0.02,
    ):
        depth = float(-relief[int(r), int(c)])
        rad = float(s * 2**0.5)
        if depth >= min_depth and not np.isnan(dtm[int(r), int(c)]) and _closed(z, r, c, rad, depth):
            out.append(Depression(float(r), float(c), rad, depth))
    return sorted(out, key=lambda d: -d.depth_m)


@dataclass
class Line:
    kind: str  # "bank" | "ditch"
    length_m: float
    pixels: np.ndarray  # (n, 2) row, col


def linear_features(dtm, sigma=3.0, min_extent=80, hi=97.0, lo=88.0, straightness=0.5, border=15):
    """Banks and ditches: ridge/valley response, hysteresis so faint stretches join strong seeds,
    skeleton, then keep long, straight components (rejects rings, tangles and noise)."""
    z = fill_nodata(dtm).astype("float64")
    H = hessian_matrix(z, sigma=sigma, order="rc", use_gaussian_derivatives=True)
    l1, l2 = hessian_matrix_eigvals(H)
    # curvature across the line must dominate curvature along it, otherwise it is a blob
    bank = np.clip(-l2, 0, None) * sigma**2 * (np.abs(l1) < 0.5 * np.abs(l2))
    ditch = np.clip(l1, 0, None) * sigma**2 * (np.abs(l2) < 0.5 * np.abs(l1))
    ok = ~ndimage.binary_dilation(np.isnan(dtm), iterations=border)
    ok[:border], ok[-border:], ok[:, :border], ok[:, -border:] = False, False, False, False
    lines: list[Line] = []
    for kind, strength in (("bank", bank), ("ditch", ditch)):
        strength = np.where(ok, strength, 0)
        pos = strength[strength > 0]
        mask = apply_hysteresis_threshold(strength, np.percentile(pos, lo), np.percentile(pos, hi))
        lab = label(skeletonize(mask), connectivity=2)
        for r in regionprops(lab):
            h0, w0, h1, w1 = r.bbox
            extent = float(np.hypot(h1 - h0, w1 - w0))
            if extent >= min_extent and extent / r.area >= straightness:
                lines.append(Line(kind, extent, r.coords))
    return sorted(lines, key=lambda l: -l.length_m)
