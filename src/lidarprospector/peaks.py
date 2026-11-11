"""Local maxima of a probability map as ranked candidate positions (no torch needed)."""

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max


def peaks_from_prob(prob: np.ndarray, min_prob: float = 0.005) -> np.ndarray:
    """(n, 2) row, col of local maxima of the smoothed map, strongest first."""
    p = ndimage.gaussian_filter(prob, 1.5)
    pk = peak_local_max(p, min_distance=6, threshold_abs=min_prob, exclude_border=False)
    return pk[np.argsort(-p[pk[:, 0], pk[:, 1]])]
