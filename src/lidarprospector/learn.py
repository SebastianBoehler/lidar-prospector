"""Learned barrow/mound detector: pixel classifier on multi-scale terrain features, scored with a
spatial split so the model is always tested on ground it never saw."""

import argparse

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage
from skimage.feature import peak_local_max
from sklearn.ensemble import HistGradientBoostingClassifier

from .detect import detect
from .features import feature_stack
from .labels import monuments
from .wcs import fetch_dtm

BUDGETS = (25, 50, 100, 200)


def rasters(dtm, x0, y1, mons, tol_m=10):
    tf = from_origin(x0, y1, 1, 1)
    tight = [m for m in mons if m["cls"] == "barrow" and m["area_ha"] <= 0.5]
    other = [m for m in mons if m not in tight]
    shape = dtm.shape

    def geom(m, v):
        return ({"type": "Polygon", "coordinates": m["rings"]}, v)

    ids = rasterize(
        [geom(m, i + 1) for i, m in enumerate(tight)], out_shape=shape, transform=tf, dtype="int32"
    )
    ign = (
        rasterize([geom(m, 1) for m in other], out_shape=shape, transform=tf, dtype="uint8")
        if other
        else np.zeros(shape, "uint8")
    )
    pos = ndimage.binary_dilation(ids > 0, iterations=2)
    dist, (ri, ci) = ndimage.distance_transform_edt(ids == 0, return_indices=True)
    near = np.where(dist <= tol_m, ids[ri, ci], 0)
    ignore = ndimage.binary_dilation(ign > 0, iterations=3) & ~pos
    return tight, ids, pos, ignore, near


def recall_at(rc, near, test_ids, k):
    got = {near[r, c] for r, c in rc[:k]} - {0}
    return len(got & test_ids) / max(len(test_ids), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=int, nargs=4, required=True)
    a = ap.parse_args()
    dtm, x0, y1 = fetch_dtm(*a.bbox)
    mons = monuments(*a.bbox)
    tight, ids, pos, ignore, near = rasters(dtm, x0, y1, mons)
    F, names = feature_stack(dtm)
    valid = ~np.isnan(dtm)
    hand, _, _ = detect(dtm)
    W = dtm.shape[1]
    split, gap = W // 2, 50
    rng = np.random.default_rng(0)
    print(
        f"{len(tight)} tight barrows; positives {pos.sum()} px; ignoring {ignore.sum()} px of other monuments"
    )

    for name, tr_cols, te_cols in (
        ("train west -> test east", (0, split - gap), (split, W)),
        ("train east -> test west", (split + gap, W), (0, split)),
    ):
        cm = np.zeros(dtm.shape, bool)
        cm[:, tr_cols[0] : tr_cols[1]] = True
        tm = np.zeros(dtm.shape, bool)
        tm[:, te_cols[0] : te_cols[1]] = True
        p_idx = np.flatnonzero(cm & pos & valid)
        n_pool = np.flatnonzero(cm & ~pos & ~ignore & valid)
        n_idx = rng.choice(n_pool, 400_000, replace=False)
        idx = np.concatenate([p_idx, n_idx])
        y = np.concatenate([np.ones(len(p_idx)), np.zeros(len(n_idx))])
        Ff = F.reshape(-1, F.shape[-1])
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.08, max_leaf_nodes=31, random_state=0
        ).fit(Ff[idx], y)
        prob = np.zeros(dtm.shape, "float32")
        te_flat = np.flatnonzero(tm & valid)
        prob.reshape(-1)[te_flat] = clf.predict_proba(Ff[te_flat])[:, 1]
        prob = ndimage.gaussian_filter(prob, 1.5)
        prob[~tm] = 0
        peaks = peak_local_max(prob, min_distance=6, threshold_abs=0.02, exclude_border=False)
        peaks = peaks[np.argsort(-prob[peaks[:, 0], peaks[:, 1]])]
        test_ids = {int(i) for i in np.unique(ids[:, te_cols[0] : te_cols[1]]) if i}
        h = sorted([c for c in hand if te_cols[0] <= c.col < te_cols[1]], key=lambda c: -c.score)
        h_rc = [(min(int(c.row), dtm.shape[0] - 1), int(c.col)) for c in h]
        vr = np.argwhere(tm & valid)
        print(
            f"\n{name}: {len(test_ids)} test barrows, {len(peaks)} learned peaks, {len(h_rc)} hand candidates"
        )
        print("  K      learned   hand-tuned   random")
        for k in BUDGETS:
            rnd = np.mean(
                [
                    len({near[r, c] for r, c in vr[rng.integers(0, len(vr), k)]} - {0} & test_ids)
                    / max(len(test_ids), 1)
                    for _ in range(30)
                ]
            )
            learned = recall_at(peaks, near, test_ids, k)
            hand_r = recall_at(h_rc, near, test_ids, k)
            print(f"  {k:<5}  {learned:7.0%}   {hand_r:9.0%}   {rnd:6.0%}")


if __name__ == "__main__":
    main()
