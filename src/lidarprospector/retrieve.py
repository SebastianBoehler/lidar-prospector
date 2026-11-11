"""Prompt-by-example search over a grid of patch embeddings (no torch needed at query time)."""

import numpy as np
from sklearn.linear_model import LogisticRegression

PATCH = 64


class Index:
    """Embeddings of one tile on a regular grid; cell (i, j) is centred at pixel (half + i*stride, ...)."""

    def __init__(self, emb: np.ndarray, stride: int = 8, patch: int = PATCH):
        self.emb = emb
        self.stride = stride
        self.half = patch // 2
        self.gh, self.gw, self.dim = emb.shape
        self.flat = emb.reshape(-1, self.dim)

    def cell_of(self, row: float, col: float) -> int:
        i = int(np.clip(round((row - self.half) / self.stride), 0, self.gh - 1))
        j = int(np.clip(round((col - self.half) / self.stride), 0, self.gw - 1))
        return i * self.gw + j

    def pixel_of(self, cell: int) -> tuple[int, int]:
        i, j = divmod(cell, self.gw)
        return self.half + i * self.stride, self.half + j * self.stride

    def top(
        self, scores: np.ndarray, k: int, sep_cells: int = 2, exclude: set[int] | None = None
    ) -> list[int]:
        """Best k cells by score, greedily suppressing neighbours so one structure gives one hit."""
        order = np.argsort(-scores)
        blocked = np.zeros(self.gh * self.gw, bool)
        picks: list[int] = []
        for c in order:
            if blocked[c] or (exclude and c in exclude):
                continue
            picks.append(int(c))
            if len(picks) == k:
                break
            i, j = divmod(int(c), self.gw)
            for ii in range(max(i - sep_cells, 0), min(i + sep_cells + 1, self.gh)):
                lo, hi = ii * self.gw + max(j - sep_cells, 0), ii * self.gw + min(j + sep_cells + 1, self.gw)
                blocked[lo:hi] = True
        return picks

    def prototype_scores(self, cells: list[int]) -> np.ndarray:
        proto = self.flat[cells].mean(0)
        return self.flat @ (proto / np.linalg.norm(proto))

    def probe_scores(
        self,
        pos_vecs: np.ndarray,
        neg_vecs: np.ndarray | None,
        rng: np.random.Generator,
        n_background: int = 3000,
    ) -> np.ndarray:
        """Linear probe: positives vs rejected examples plus random background (structures are rare, so
        random places are almost all negative). Learns the direction in embedding space that separates
        'like these' from the rest, which beats the mean of the examples because it also learns what to
        ignore (field boundaries, tracks)."""
        bg = self.flat[rng.integers(0, len(self.flat), n_background)]
        neg = bg if neg_vecs is None or len(neg_vecs) == 0 else np.concatenate([neg_vecs, bg])
        x = np.concatenate([pos_vecs, neg])
        y = np.r_[np.ones(len(pos_vecs)), np.zeros(len(neg))]
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(x, y)
        return self.flat @ clf.coef_[0]

    def classifier_scores(
        self, pos: list[int], neg: list[int], rng: np.random.Generator, n_background: int = 3000
    ) -> np.ndarray:
        """probe_scores for examples given as cells of this index."""
        return self.probe_scores(self.flat[pos], self.flat[neg] if neg else None, rng, n_background)
