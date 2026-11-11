"""Multi-scale terrain descriptors per pixel: band-pass relief (mounds +, ditches -), Hessian
eigenvalues (blob / ridge / valley shape at each scale), slope and local roughness."""

import numpy as np
from scipy import ndimage
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

RELIEF_SIGMAS = (1.5, 3.0, 6.0, 12.0)
HESSIAN_SIGMAS = (2.0, 4.0, 8.0)


def fill_nodata(dtm):
    nan = np.isnan(dtm)
    if not nan.any():
        return dtm
    _, (ri, ci) = ndimage.distance_transform_edt(nan, return_indices=True)
    return dtm[ri, ci]


def feature_stack(dtm):
    """(H, W, F) float32 plus the feature names."""
    z = fill_nodata(dtm).astype("float64")
    chans, names = [], []
    for s in RELIEF_SIGMAS:
        chans.append(ndimage.gaussian_filter(z, s) - ndimage.gaussian_filter(z, 3 * s))
        names.append(f"relief{s}")
    for s in HESSIAN_SIGMAS:
        H = hessian_matrix(z, sigma=s, order="rc", use_gaussian_derivatives=True)
        l1, l2 = hessian_matrix_eigvals(H)  # l1 >= l2
        chans += [l1 * s**2, l2 * s**2]
        names += [f"eig1_{s}", f"eig2_{s}"]
    gy, gx = np.gradient(ndimage.gaussian_filter(z, 2.0))
    slope = np.hypot(gx, gy)
    chans.append(slope)
    names.append("slope2")
    chans.append(np.sqrt(np.maximum(ndimage.uniform_filter(slope**2, 21), 0.0)))
    names.append("rough10")
    chans.append(ndimage.gaussian_filter(z, 25) - ndimage.gaussian_filter(z, 75))
    names.append("regional")
    out = np.stack(chans, axis=-1).astype("float32")
    out[np.isnan(dtm)] = np.nan
    return out, names
