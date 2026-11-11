"""Local click-to-find viewer: show a terrain tile, click examples, find similar structures.

uv run python -m lidarprospector.serve --bbox 410800 140800 413800 143800
    uv run python -m lidarprospector.serve --source geotiff:dem.tif --licence "..." --attribution "..." \\
        --bbox XMIN YMIN XMAX YMAX
"""

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy import ndimage

from .embed import Encoder, embed_grid, relief_channels
from .labels import monuments
from .learn import rasters
from .retrieve import Index
from .terrain import hillshade
from .wcs import fetch_dtm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STATIC = Path(__file__).parent / "static"


class Tile:
    """A terrain window as an image plus patch embeddings. `known` are optional (row, col) positions of
    known structures to overlay; terrain without labels simply has none."""

    def __init__(self, dtm, known, encoder_path, device):
        self.shape = dtm.shape
        buf = io.BytesIO()
        # fixed 0-1 scale: automatic scaling returns NaN (all-transparent image) when the terrain has nodata
        plt.imsave(buf, hillshade(dtm), cmap="gray", vmin=0.0, vmax=1.0, format="png")
        self.png = buf.getvalue()
        model = Encoder().to(device)
        model.load_state_dict(torch.load(encoder_path, map_location=device))
        self.index = Index(embed_grid(model, relief_channels(dtm), device=device))
        self.known = known
        self.rng = np.random.default_rng(0)

    def search(self, pos, neg, k):
        idx = self.index
        pos_c = [idx.cell_of(r, c) for r, c in pos]
        neg_c = [idx.cell_of(r, c) for r, c in neg]
        scores = idx.probe_scores(idx.flat[pos_c], idx.flat[neg_c] if neg_c else None, self.rng)
        hide = set(pos_c + neg_c)
        top = idx.top(scores, k, exclude=hide)
        return [{"row": r, "col": c} for r, c in (idx.pixel_of(t) for t in top)]


def handler(tile: Tile):
    class H(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/tile.png":
                self._send(tile.png, "image/png")
            elif self.path == "/info":
                body = {"height": tile.shape[0], "width": tile.shape[1], "known": tile.known}
                self._send(json.dumps(body).encode(), "application/json")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/search":
                return self.send_error(404)
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not req["pos"]:
                return self.send_error(400, "need at least one positive example")
            found = tile.search(req["pos"], req["neg"], int(req.get("k", 30)))
            self._send(json.dumps(found).encode(), "application/json")

        def log_message(self, *args):
            pass

    return H


def england_tile(bbox, encoder_path, device) -> Tile:
    dtm, x0, y1 = fetch_dtm(*bbox)
    tight, ids, *_ = rasters(dtm, x0, y1, monuments(*bbox))
    n = len(tight)
    known = np.array(ndimage.center_of_mass(ids > 0, ids, range(1, n + 1))).tolist() if n else []
    return Tile(dtm, known, encoder_path, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, required=True)
    ap.add_argument(
        "--source", default="england", help="england (with scheduled-monument overlay) or any scan source"
    )
    ap.add_argument("--licence")
    ap.add_argument("--attribution")
    ap.add_argument("--epsg", type=int)
    ap.add_argument("--encoder", type=Path, default=Path("output/encoder.pt"))
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("loading terrain and embedding the tile ...", flush=True)
    if a.source == "england":
        tile = england_tile(tuple(int(v) for v in a.bbox), a.encoder, device)
    else:
        from .scan import build_source

        args = argparse.Namespace(source=a.source, licence=a.licence, attribution=a.attribution, epsg=a.epsg,
                                  lonlat=None)  # fmt: skip
        src = build_source(args)
        print(f"data: {src.name}; {src.licence}; {src.attribution}", flush=True)
        tile = Tile(src.fetch(tuple(a.bbox), cell=1.0).z, [], a.encoder, device)
    print(f"ready: http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(tile)).serve_forever()


if __name__ == "__main__":
    main()
