"""Seam-safe tiling: process big areas in bounded memory and get the answer one big image would give.

* Each tile is computed with a *halo* of context (>= the widest filter), so core pixels are exact.
* Every detection is owned by the one tile whose *core* contains it: no duplicates, no gaps.
* Line fragments that meet at a seam are joined (union-find) before any length/shape filtering, so a
  bank crossing three tiles is one line and is not rejected as three short pieces.
* A resolution pyramid finds structures bigger than a tile's fine-scale filters can see.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .shapes import Line


@dataclass(frozen=True)
class Tile:
    index: tuple[int, int]
    row0: int  # core origin, global pixel rows/cols
    col0: int
    rows: int
    cols: int
    halo: int

    def owns(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Mask of global positions inside this tile's core."""
        return (
            (rows >= self.row0)
            & (rows < self.row0 + self.rows)
            & (cols >= self.col0)
            & (cols < self.col0 + self.cols)
        )


class TileGrid:
    """Tiles over a bbox (xmin, ymin, xmax, ymax in CRS metres) on a global pixel grid anchored at
    (xmin, ymax). `core` and `halo` are in pixels; keep them multiples of the largest pyramid scale."""

    def __init__(self, bbox, cell: float, core: int, halo: int):
        self.xmin, self.ymin, self.xmax, self.ymax = bbox
        self.cell, self.core, self.halo = cell, core, halo
        self.width = math.ceil((self.xmax - self.xmin) / cell)
        self.height = math.ceil((self.ymax - self.ymin) / cell)

    def __iter__(self) -> Iterator[Tile]:
        for ti, r in enumerate(range(0, self.height, self.core)):
            for tj, c in enumerate(range(0, self.width, self.core)):
                yield Tile(
                    (ti, tj), r, c, min(self.core, self.height - r), min(self.core, self.width - c), self.halo
                )

    def __len__(self) -> int:
        return math.ceil(self.height / self.core) * math.ceil(self.width / self.core)

    def window_bbox(self, t: Tile) -> tuple[float, float, float, float]:
        """CRS bbox of the tile including its halo (may extend beyond the scan area)."""
        h, c = t.halo, self.cell
        return (
            self.xmin + (t.col0 - h) * c,
            self.ymax - (t.row0 + t.rows + h) * c,
            self.xmin + (t.col0 + t.cols + h) * c,
            self.ymax - (t.row0 - h) * c,
        )


def downsample(z: np.ndarray, s: int) -> np.ndarray:
    """NaN-aware block mean by an integer factor (edges that do not fill a block are dropped)."""
    if s == 1:
        return z
    h, w = (z.shape[0] // s) * s, (z.shape[1] // s) * s
    blocks = z[:h, :w].reshape(h // s, s, w // s, s)
    n = np.sum(~np.isnan(blocks), axis=(1, 3))
    total = np.nansum(blocks, axis=(1, 3))
    return np.where(n > 0, total / np.maximum(n, 1), np.nan).astype(np.float32)


def scales_for(
    min_size_m: float, max_size_m: float, cell: float = 1.0, widest_px: float = 30.0
) -> tuple[int, ...]:
    """Pyramid factors (powers of two) whose detectors cover structure widths in [min, max] metres.
    At factor s the fine-scale filters respond to widths of roughly 3*s*cell .. widest_px*s*cell."""
    out = []
    s = 1
    while s * cell * 3 <= max_size_m:
        if s * cell * widest_px >= min_size_m:
            out.append(s)
        s *= 2
    return tuple(out) or (1,)


def required_halo(scales: tuple[int, ...]) -> int:
    """Context (base pixels) a tile needs so its core is exact. At pyramid factor s the line finder
    reaches ~12 s px (Hessian) plus its 15 s px edge mask; the pit finder reaches 120 px (30 px Gaussian
    x 4) and the mound network ~100 px at the base scale."""
    return max(160, 30 * max(scales))


class _DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def merge_line_fragments(fragments: list[tuple[tuple, Line]], seam_px: int = 3) -> list[Line]:
    """Join fragments (key, Line with *global* pixels, already clipped to their tile core).

    Fragments with the same key came from one component in one tile and are always joined. Fragments
    of the same kind from different tiles are joined where their pixels touch (8-neighbourhood); only
    pixels near a core boundary can touch, so only those are indexed (`seam_px`, see `clip_fragments`)."""
    ds = _DisjointSet(len(fragments))
    first: dict[tuple, int] = {}
    for i, (key, _) in enumerate(fragments):
        if key in first:
            ds.union(i, first[key])
        else:
            first[key] = i
    at: dict[tuple[str, int, int], int] = {}
    for i, (_, line) in enumerate(fragments):
        for r, c in line.pixels:
            at[(line.kind, int(r), int(c))] = i
    for i, (_, line) in enumerate(fragments):
        for r, c in line.pixels:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    j = at.get((line.kind, int(r) + dr, int(c) + dc))
                    if j is not None and j != i:
                        ds.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(fragments)):
        groups.setdefault(ds.find(i), []).append(i)
    merged = []
    for members in groups.values():
        pixels = np.concatenate([fragments[i][1].pixels for i in members])
        merged.append(Line(fragments[members[0]][1].kind, 0.0, pixels))
    return merged


def clip_fragments(tile: Tile, lines: list[Line], to_global) -> list[tuple[tuple, Line]]:
    """Move a tile's fragments to global pixels and keep only the part inside the core."""
    out = []
    for k, line in enumerate(lines):
        g = to_global(line.pixels)
        keep = tile.owns(g[:, 0], g[:, 1])
        if keep.any():
            out.append(((tile.index, k), Line(line.kind, 0.0, g[keep])))
    return out
