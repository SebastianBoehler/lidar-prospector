"""Bare-earth grid from classified ground returns, plus visualisations that make
low earthworks readable: local relief model and multi-direction hillshade."""

import numpy as np
from scipy import ndimage

GROUND = 2


def ground_dtm(x, y, z, cls, cell: float = 1.0, max_gap_cells: int = 6):
    """Mean ground elevation per cell. Gaps up to `max_gap_cells` are filled from the
    nearest ground cell; larger gaps (water, buildings) stay NaN."""
    g = cls == GROUND
    x, y, z = x[g], y[g], z[g]
    x0, y0 = x.min(), y.min()
    ix = ((x - x0) / cell).astype(int)
    iy = ((y - y0) / cell).astype(int)
    shape = (iy.max() + 1, ix.max() + 1)
    flat = iy * shape[1] + ix
    n = np.bincount(flat, minlength=shape[0] * shape[1])
    s = np.bincount(flat, weights=z, minlength=shape[0] * shape[1])
    dtm = np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(shape)

    empty = np.isnan(dtm)
    dist, (ri, ci) = ndimage.distance_transform_edt(empty, return_indices=True)
    filled = dtm[ri, ci]
    filled[dist > max_gap_cells] = np.nan
    return filled[::-1], (x0, y0, cell)  # row 0 = north


def local_relief(dtm, sigma_m: float = 15.0, cell: float = 1.0):
    """DTM minus its smoothed trend: mounds positive, ditches negative (metres)."""
    valid = ~np.isnan(dtm)
    d = np.where(valid, dtm, 0.0)
    s = ndimage.gaussian_filter(d, sigma_m / cell)
    w = ndimage.gaussian_filter(valid.astype(float), sigma_m / cell)
    trend = np.where(w > 0.05, s / np.maximum(w, 1e-9), np.nan)
    return np.where(valid, dtm - trend, np.nan)


def hillshade(dtm, cell: float = 1.0, azimuths=(315, 45, 135, 225), altitude=40.0, z_exag=3.0):
    """Mean of several illumination directions so no linear feature is hidden."""
    filled = np.where(np.isnan(dtm), np.nanmean(dtm), dtm) * z_exag
    gy, gx = np.gradient(filled, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    alt = np.radians(altitude)
    shades = []
    for az in azimuths:
        a = np.radians(az)
        shades.append(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(a - aspect))
    out = np.clip(np.mean(shades, axis=0), 0, 1)
    out[np.isnan(dtm)] = np.nan
    return out
