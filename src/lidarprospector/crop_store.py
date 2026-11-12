"""Terrain crops from an on-disk corpus, so training never holds the corpus in RAM or on the GPU.

Each terrain's relief channels are computed once and stored as float16 memory maps. Sampling draws half the
crops uniformly over all terrain and half weighted by local relief energy (so structured ground is seen
often), reading only the crop from disk. A corpus of any size costs a few MB of RAM.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy import ndimage

from .embed import relief_channels

STRIDE = 16


class CropStore:
    def __init__(self, files: list[Path], crop: int):
        self.maps = [np.load(f, mmap_mode="r") for f in files]
        coords, energy = [], []
        for t, m in enumerate(self.maps):
            h, w = m.shape[1:]
            ys, xs = np.mgrid[0 : h - crop : STRIDE, 0 : w - crop : STRIDE]
            e = ndimage.uniform_filter(np.abs(np.asarray(m[0], dtype=np.float32)), crop)
            coords.append(np.c_[np.full(ys.size, t), ys.ravel(), xs.ravel()].astype(np.int32))
            energy.append(e[ys + crop // 2, xs + crop // 2].ravel() + 1e-3)
        self.coords = np.concatenate(coords)
        p = np.concatenate(energy).astype(np.float64)
        self.cdf = np.cumsum(p / p.sum())
        self.crop = crop

    @classmethod
    def build(cls, items: list[tuple[str, Callable[[], np.ndarray]]], cache: Path, crop: int = 128):
        """items: (unique key, loader returning a terrain array). Existing cache files are reused."""
        cache.mkdir(parents=True, exist_ok=True)
        files = []
        for key, load in items:
            f = cache / f"{key}.npy"
            if not f.exists():
                np.save(f, relief_channels(load()).astype(np.float16))
            files.append(f)
        return cls(files, crop)

    def __len__(self) -> int:
        return len(self.maps)

    def sample(self, n: int, crop: int, rng: np.random.Generator) -> np.ndarray:
        """(n, 2, crop, crop) float32 crops: half uniform over the corpus, half energy-weighted."""
        if crop != self.crop:
            raise ValueError(f"store built for crop={self.crop}, asked for {crop}")
        half = n // 2
        idx = np.concatenate(
            [rng.integers(0, len(self.coords), half), np.searchsorted(self.cdf, rng.random(n - half))]
        )
        idx = np.minimum(idx, len(self.coords) - 1)
        out = np.empty((n, 2, crop, crop), np.float32)
        for i, k in enumerate(idx):
            t, y, x = self.coords[k]
            out[i] = self.maps[t][:, y : y + crop, x : x + crop]
        return out
