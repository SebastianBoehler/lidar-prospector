"""Supervised ceiling for one object type: could a model trained on labelled examples find it at all?

Leave-one-tile-out over the Dartmoor tiles with a U-Net trained for that single type. If this is also near
chance, the limit is the data (few, imprecise polygons, subtle relief), not the prompting method.
"""

import argparse

import numpy as np
from rasterio import Affine
from rasterio.features import rasterize
from scipy import ndimage

from .crossval import landscape_tiles
from .embed import relief_channels
from .features_torch import pick_device
from .labels import monuments
from .models_bench import train_unet
from .mounds import predict_dense
from .peaks import peaks_from_prob
from .prompt_types import hit_counts, polygon_ids, typed_objects
from .wcs import fetch_dtm

TOP = 100


def load(bbox, kind):
    dtm = fetch_dtm(*bbox)[0]
    centres, rings = typed_objects(bbox)
    shape = dtm.shape
    pos = np.zeros(shape, bool)
    for r in rings.get(kind, []):
        pos |= rasterize([({"type": "Polygon", "coordinates": [r.tolist()]}, 1)], out_shape=shape,
                         transform=Affine.identity(), dtype="uint8") > 0  # fmt: skip
    others = [{"type": "Polygon", "coordinates": [np.c_[np.array(m["rings"][0])[:, 0] - bbox[0],
                                                       bbox[3] - np.array(m["rings"][0])[:, 1]].tolist()]}
              for m in monuments(*bbox)]  # fmt: skip
    ign = rasterize([(g, 1) for g in others], out_shape=shape, transform=Affine.identity(), dtype="uint8") > 0
    ignore = ndimage.binary_dilation(ign, iterations=3) & ~ndimage.binary_dilation(pos, iterations=2)
    return {"dtm": dtm, "chan": relief_channels(dtm), "pos": ndimage.binary_dilation(pos, iterations=2),
            "ignore": ignore, "n": len(rings.get(kind, [])), "rings": rings.get(kind, [])}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=["hut circle", "cairn", "stone setting"])
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    device = pick_device()
    data = [load(b, a.kind) for name, b in landscape_tiles() if name == "Dartmoor"]
    found = total = 0
    for i, test in enumerate(data):
        if test["n"] == 0:
            continue
        train = [t for j, t in enumerate(data) if j != i and t["n"] > 0]
        model = train_unet(train, None, a.steps, device, seed=i)
        pk = peaks_from_prob(predict_dense(model, test["chan"], device))[:TOP]
        ids = {a.kind: polygon_ids(test["rings"], test["dtm"].shape)}
        f = hit_counts(pk, ids)[a.kind]
        found, total = found + f, total + test["n"]
        print(f"fold {i}: {f}/{test['n']} {a.kind} polygons contain a top-{TOP} candidate", flush=True)
    rate = f"{found}/{total} = {found / total:.0%}"
    print(f"\nSUPERVISED CEILING ({a.kind}, leave-one-tile-out, Dartmoor): {rate}")


if __name__ == "__main__":
    main()
