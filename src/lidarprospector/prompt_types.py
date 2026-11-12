"""Is a prompt really steering the model to a *type*, or is it just a generic bump finder?

Five object types, labelled in 13 English tiles (single monuments, <= 0.5 ha). For each feature extractor,
type and prompt size k, examples come from OTHER tiles, a probe is fitted, and the held-out tile is ranked:
  own-type recall   = share of that type's scheduled polygons containing a top-100 candidate
  off-type recall   = the same for the other types (high means the prompt is not discriminating)
No model is trained for any type. Random placement is the control.
"""

import argparse
import re
from collections import defaultdict

import numpy as np
import torch
from rasterio import Affine
from rasterio.features import rasterize
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

from .crossval import landscape_tiles, tiles
from .features_torch import pick_device
from .labels import monuments
from .mae import mae_encoder_extractor, mae_extractor
from .promptable import (
    Extractor,
    fit_probe,
    hand_extractor,
    relief_band,
    score_maps,
    ssl_extractor,
    template_scores,
    unet_extractor,
)
from .wcs import fetch_dtm

TYPES = ("round barrow", "cairn", "hut circle", "small enclosure", "stone setting")
KS = (1, 3, 10)
TRIALS = 3
TOP = 100  # candidates per tile


def object_type(name: str | None) -> str | None:
    n = (name or "").lower()
    if re.search(r"hut circle|hut-circle|roundhouse|round house", n):
        return "hut circle"
    if re.search(r"stone row|stone circle|standing stone|menhir|stone alignment", n):
        return "stone setting"
    if re.search(r"\bcairn", n):
        return "cairn"
    if "barrow" in n and "long barrow" not in n:
        return "round barrow"
    if "enclosure" in n:
        return "small enclosure"
    return None


def typed_objects(bbox):
    """type -> ((n, 2) pixel (row, col) centres, [outer rings in pixel (col, row)]) of single monuments."""
    centres, rings = defaultdict(list), defaultdict(list)
    for m in monuments(*bbox):
        t = object_type(m["name"])
        if t and m["area_ha"] is not None and m["area_ha"] <= 0.5:
            ring = np.array(m["rings"][0])
            col, row = ring[:, 0] - bbox[0], bbox[3] - ring[:, 1]
            centres[t].append([row.mean(), col.mean()])
            rings[t].append(np.c_[col, row])
    return {t: np.array(v) for t, v in centres.items()}, dict(rings)


def typed_centres(bbox) -> dict[str, np.ndarray]:
    return typed_objects(bbox)[0]


def polygon_ids(rings, shape, grow: int = 3) -> np.ndarray:
    """Raster of 1-based monument ids over the scheduled polygons, grown by `grow` pixels."""
    shapes = [({"type": "Polygon", "coordinates": [r.tolist()]}, i + 1) for i, r in enumerate(rings)]
    ids = rasterize(shapes, out_shape=shape, transform=Affine.identity(), dtype="int32")
    dist, (ri, ci) = ndimage.distance_transform_edt(ids == 0, return_indices=True)
    return np.where(dist <= grow, ids[ri, ci], 0)


def neighbourhood(feats, centres):
    """(n, 9, D): the 3x3 pixels around each centre, so a single example gives nine training vectors."""
    h, w, _ = feats.shape
    r = np.clip(centres[:, 0].round().astype(int), 1, h - 2)
    c = np.clip(centres[:, 1].round().astype(int), 1, w - 2)
    return np.stack([feats[r + dr, c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=1)


def collect(ex: Extractor, boxes, labels, rng):
    """Phase 1: per tile, the feature vectors at every labelled centre and a pool of background pixels."""
    vec, bg = [], []
    for b, lab in zip(boxes, labels, strict=True):
        feats = ex.dense(fetch_dtm(*b)[0])
        vec.append({t: neighbourhood(feats, c) for t, c in lab.items()})
        allc = np.concatenate(list(lab.values())) if lab else np.zeros((0, 2))
        h, w, _ = feats.shape
        pts = rng.integers(20, [h - 20, w - 20], size=(4000, 2))
        if len(allc):
            pts = pts[cKDTree(allc).query(pts)[0] > 20]
        bg.append(feats[pts[:, 0], pts[:, 1]])
    return vec, bg


def peaks(score: np.ndarray) -> np.ndarray:
    m = ndimage.gaussian_filter(score, 1.0)
    return peak_local_max(m, min_distance=3, num_peaks=TOP, exclude_border=False) * 2 + 1  # pooled by 2


def hit_counts(pk, ids):
    """Monuments (per type) that contain at least one top-K candidate: `ids` maps type -> polygon raster."""
    out = {}
    for t, raster in ids.items():
        found = raster[np.clip(pk[:, 0], 0, raster.shape[0] - 1), np.clip(pk[:, 1], 0, raster.shape[1] - 1)]
        out[t] = len(set(found[found > 0].tolist()))
    return out


def evaluate(ex: Extractor, boxes, labels, polys, rng, device):
    vec, bg = collect(ex, boxes, labels, rng)
    allbg = np.concatenate(bg)
    mean, std = allbg.mean(0), allbg.std(0) + 1e-6
    found = defaultdict(float)  # (k, prompt type, eval type) -> hits
    total = defaultdict(float)
    rand = defaultdict(float)
    for i, b in enumerate(boxes):
        if not labels[i]:
            continue
        feats = ex.dense(fetch_dtm(*b)[0])
        h, w, d = feats.shape
        ids = {t: polygon_ids(polys[i][t], (h, w)) for t in labels[i]}
        ft = torch.from_numpy(((feats - mean) / std).reshape(-1, d)).to(device)
        neg = np.concatenate([x for j, x in enumerate(bg) if j != i])
        neg = (neg - mean) / std
        models = []
        for t in TYPES:
            pool = [v[t] for j, v in enumerate(vec) if j != i and t in v]
            pool = np.concatenate(pool) if pool else np.zeros((0, 9, d))
            if len(pool) == 0:
                continue
            for k in KS:
                for _ in range(TRIALS):
                    pick = pool[rng.choice(len(pool), min(k, len(pool)), replace=False)].reshape(-1, d)
                    models.append((k, t, *fit_probe((pick - mean) / std, neg)))
        for lo in range(0, len(models), 15):
            part = models[lo : lo + 15]
            s = score_maps(
                ft, np.stack([m[2] for m in part]), np.array([m[3] for m in part], np.float32), (h, w)
            )
            for (k, t, _, _), sm in zip(part, s.cpu().numpy(), strict=True):
                for u, n in hit_counts(peaks(sm), ids).items():
                    found[(k, t, u)] += n
        for _ in range(TRIALS * len(KS) * 5):
            pk = rng.integers(0, [h, w], size=(TOP, 2))
            for u, n in hit_counts(pk, ids).items():
                rand[u] += n / (TRIALS * len(KS) * 5)
        for u, c in labels[i].items():
            for k in KS:
                for t in TYPES:
                    total[(k, t, u)] += len(c) * TRIALS
    return found, total, rand


PATCH = 31  # metres around an example that the template covers


def evaluate_template(boxes, labels, polys, rng, device):
    """Template matching: no features, no probe; the examples' raw relief patches are the prompt."""
    reliefs = [relief_band(fetch_dtm(*b)[0]) for b in boxes]
    half = PATCH // 2
    patches = []
    for rel, lab in zip(reliefs, labels, strict=True):
        got = {}
        for t, c in lab.items():
            r = np.clip(c[:, 0].round().astype(int), half, rel.shape[0] - half - 1)
            q = np.clip(c[:, 1].round().astype(int), half, rel.shape[1] - half - 1)
            got[t] = np.stack(
                [rel[a - half : a + half + 1, b - half : b + half + 1] for a, b in zip(r, q, strict=True)]
            )
        patches.append(got)
    found, total, rand = defaultdict(float), defaultdict(float), defaultdict(float)
    for i in range(len(boxes)):
        if not labels[i]:
            continue
        rel = reliefs[i]
        ids = {t: polygon_ids(polys[i][t], rel.shape) for t in labels[i]}
        for t in TYPES:
            pool = [p[t] for j, p in enumerate(patches) if j != i and t in p]
            if not pool:
                continue
            pool = np.concatenate(pool)
            for k in KS:
                for _ in range(TRIALS):
                    pick = pool[rng.choice(len(pool), min(k, len(pool)), replace=False)]
                    sm = template_scores(rel, pick, device).cpu().numpy()
                    for u, n in hit_counts(peaks(sm), ids).items():
                        found[(k, t, u)] += n
        for _ in range(TRIALS * len(KS) * 5):
            pk = rng.integers(0, rel.shape, size=(TOP, 2))
            for u, n in hit_counts(pk, ids).items():
                rand[u] += n / (TRIALS * len(KS) * 5)
        for u, c in labels[i].items():
            for k in KS:
                for t in TYPES:
                    total[(k, t, u)] += len(c) * TRIALS
    return found, total, rand


def report(name, found, total, rand):
    n_eval = {}
    for u in TYPES:
        n_eval[u] = sum(v for (k, t, uu), v in total.items() if uu == u and k == 10 and t == u) / TRIALS
    print(f"\n=== {name}")
    print("own-type recall (a candidate inside the polygon) at 100 candidates per 9 km2 tile, by prompt size")
    print(f"{'type':<18}{'n':>5}" + "".join(f"{f'k={k}':>8}" for k in KS) + f"{'random':>9}")
    for t in TYPES:
        n = n_eval[t]
        if n == 0:
            continue
        row = [found[(k, t, t)] / total[(k, t, t)] if total[(k, t, t)] else float("nan") for k in KS]
        print(f"{t:<18}{int(n):>5}" + "".join(f"{v:>8.0%}" for v in row) + f"{rand[t] / n:>9.0%}")
    print("\nwhere do k=10 prompts land? rows: prompt type; columns: type of monument hit (recall)")
    corner = "prompt / hit"
    print(f"{corner:<18}" + "".join(f"{u[:11]:>13}" for u in TYPES))
    for t in TYPES:
        cells = [found[(10, t, u)] / total[(10, t, u)] if total[(10, t, u)] else float("nan") for u in TYPES]
        print(f"{t:<18}" + "".join(f"{v:>13.0%}" for v in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mae-weights", default="output/mae.pt")
    ap.add_argument("--unet-weights", default="output/unet.pt", help="weights for the U-Net feature source")
    ap.add_argument(
        "--only", choices=["hand", "unet", "ssl", "mae", "maeenc", "template"], help="run a single method"
    )
    a = ap.parse_args()
    boxes = list(tiles()) + [b for _, b in landscape_tiles()]
    objects = [typed_objects(b) for b in boxes]
    labels = [o[0] for o in objects]
    polys = [o[1] for o in objects]
    counts = {t: sum(len(lab.get(t, [])) for lab in labels) for t in TYPES}
    print("single monuments per type across", len(boxes), "tiles:", counts)
    device = pick_device()
    makers = {
        "hand": hand_extractor,
        "unet": lambda: unet_extractor(a.unet_weights),
        "ssl": ssl_extractor,
        "mae": lambda: mae_extractor(a.mae_weights),
        "maeenc": lambda: mae_encoder_extractor(a.mae_weights),
    }
    for name in [a.only] if a.only else ["hand", "unet", "ssl", "template"]:
        rng = np.random.default_rng(0)
        if name == "template":
            report(
                "template matching on raw terrain (no features, no learning)",
                *evaluate_template(boxes, labels, polys, rng, device),
            )
        else:
            ex = makers[name]()
            report(ex.name, *evaluate(ex, boxes, labels, polys, rng, device))


if __name__ == "__main__":
    main()
