"""Area scan: tile an area with halos, download tiles in parallel with compute, run the detectors, and
merge across seams. Memory is bounded by the plan; a MemoryGate waits if other apps take RAM meanwhile."""

import math
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from pyproj import Transformer

from .peaks import peaks_from_prob
from .runtime import MemoryGate, Plan
from .shapes import DEFAULT_THRESHOLDS, Line, depressions, keep_line, line_fragments
from .tiling import (
    Tile,
    TileGrid,
    clip_fragments,
    downsample,
    merge_line_fragments,
    required_halo,
)


@dataclass
class ScanConfig:
    cell: float = 1.0  # working resolution in metres; the detectors are tuned for 1 m
    scales: tuple[int, ...] = (1,)  # pyramid factors for line features (see tiling.scales_for)
    min_extent_m: float = 80.0  # shortest line reported
    thresholds: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    mound_fn: Callable[[np.ndarray], np.ndarray] | None = None  # dtm tile -> barrow probability map
    min_mound_prob: float = 0.3
    # Probabilities from a loss that up-weights positives are not calibrated, so rank instead: keep the
    # best N per km2 over the whole scan (the benchmark's K=200 per 9 km2 is ~22). None keeps all.
    max_mounds_per_km2: float | None = 22.0


@dataclass
class ScanResult:
    x0: float
    y1: float
    cell: float
    epsg: int
    source: str
    licence: str
    attribution: str
    mounds: list[tuple[float, float, float]] = field(default_factory=list)  # row, col, prob (base px)
    pits: list[tuple[float, float, float, float]] = field(default_factory=list)  # row, col, radius_m, depth_m
    lines: list[tuple[Line, int]] = field(default_factory=list)  # (line in pixels of its scale, scale)
    tiles: int = 0
    seconds_waiting_for_data: float = 0.0
    seconds_computing: float = 0.0

    def xy(self, row: float, col: float, scale: int = 1) -> tuple[float, float]:
        half = scale / 2
        return self.x0 + (col * scale + half) * self.cell, self.y1 - (row * scale + half) * self.cell

    def to_geojson(self) -> dict:
        to_ll = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        feats = []

        def feature(kind, geometry, **props):
            feats.append({"type": "Feature", "properties": {"kind": kind, **props}, "geometry": geometry})

        def point(kind, r, c, **props):
            lon, lat = to_ll.transform(*self.xy(r, c))
            feature(kind, {"type": "Point", "coordinates": [lon, lat]}, **props)

        for r, c, p in self.mounds:
            point("mound", r, c, score=round(float(p), 3))
        for r, c, rad, depth in self.pits:
            point("pit", r, c, radius_m=round(rad, 1), depth_m=round(depth, 2))
        for line, s in self.lines:
            step = max(1, 3 // s)
            pts = [list(to_ll.transform(*self.xy(r, c, s))) for r, c in line.pixels[::step]]
            length = round(line.extent * s * self.cell)
            feature(line.kind, {"type": "MultiPoint", "coordinates": pts}, length_m=length, scale=s)
        meta = {"source": self.source, "licence": self.licence, "attribution": self.attribution}
        return {"type": "FeatureCollection", "features": feats, "properties": meta}


def _process(tile: Tile, z: np.ndarray, cfg: ScanConfig):
    """Detections for one halo'd tile: points owned by the core, line fragments clipped to it."""
    h = tile.halo
    off = np.array([tile.row0 - h, tile.col0 - h])
    mounds, pits = [], []
    if cfg.mound_fn is not None:
        pk = peaks_from_prob(prob := cfg.mound_fn(z))
        g = pk + off
        own = tile.owns(g[:, 0], g[:, 1])
        mounds = [
            (r, c, prob[p[0], p[1]])
            for (r, c), p in zip(g[own], pk[own], strict=True)
            if prob[p[0], p[1]] >= cfg.min_mound_prob
        ]
    for d in depressions(z):
        if tile.owns(np.array([d.row + off[0]]), np.array([d.col + off[1]]))[0]:
            pits.append((d.row + off[0], d.col + off[1], d.radius_m, d.depth_m))
    frags = {}
    for s in cfg.scales:
        zs = downsample(z, s)
        ts = Tile(
            tile.index,
            tile.row0 // s,
            tile.col0 // s,
            math.ceil(tile.rows / s),
            math.ceil(tile.cols / s),
            h // s,
        )
        local = line_fragments(zs, thresholds=cfg.thresholds)
        shift = np.array([ts.row0 - ts.halo, ts.col0 - ts.halo])
        frags[s] = clip_fragments(ts, local, lambda px, shift=shift: px + shift)
    return mounds, pits, frags


def scan(
    source, bbox, plan: Plan, cfg: ScanConfig | None = None, log=print, gate: MemoryGate | None = None
) -> ScanResult:
    cfg = cfg or ScanConfig()
    align = max(cfg.scales)
    halo = math.ceil(max(plan.halo, required_halo(cfg.scales)) / align) * align
    core = max(align, (plan.tile_core // align) * align)
    grid = TileGrid(bbox, cfg.cell, core, halo)
    gate = gate or MemoryGate()
    need = int((core + 2 * halo) ** 2 * plan.bytes_per_px)
    res = ScanResult(
        grid.xmin, grid.ymax, cfg.cell, source.epsg, source.name, source.licence, source.attribution
    )
    if source.native_cell > 2 * cfg.cell:
        res_m, small = source.native_cell, 4 * source.native_cell
        log(f"warning: {res_m:.0f} m source; features under ~{small:.0f} m are lost")
    log(f"{len(grid)} tiles of {core}+{halo} px halo; source: {source.name} ({source.licence})")
    all_frags: dict[int, list] = {s: [] for s in cfg.scales}
    tiles = iter(grid)
    queue: deque = deque()
    with ThreadPoolExecutor(plan.io_workers) as io:

        def submit():
            t = next(tiles, None)
            if t is not None:
                queue.append((t, io.submit(source.fetch, grid.window_bbox(t), cfg.cell)))

        for _ in range(plan.io_workers + 1):
            submit()
        done = 0
        while queue:
            tile, fut = queue.popleft()
            submit()  # keep the pipe full while this tile computes
            t0 = time.perf_counter()
            dem = fut.result()
            res.seconds_waiting_for_data += time.perf_counter() - t0
            t0 = time.perf_counter()
            with gate.admit(need):
                mounds, pits, frags = _process(tile, dem.z, cfg)
            res.seconds_computing += time.perf_counter() - t0
            res.mounds += mounds
            res.pits += pits
            for s, f in frags.items():
                all_frags[s] += f
            done += 1
            log(f"tile {done}/{len(grid)} {tile.index}: {len(mounds)} mounds, {len(pits)} pits")
    for s, frags in all_frags.items():
        for line in merge_line_fragments(frags):
            if keep_line(line, cfg.min_extent_m / (cfg.cell * s)):
                res.lines.append((line, s))
    if cfg.max_mounds_per_km2 is not None:
        area_km2 = grid.width * grid.height * cfg.cell**2 / 1e6
        keep = max(1, round(cfg.max_mounds_per_km2 * area_km2))
        res.mounds = sorted(res.mounds, key=lambda m: -m[2])[:keep]
    res.tiles = len(grid)
    return res
