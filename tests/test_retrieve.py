import numpy as np

from lidarprospector.retrieve import Index


def make_index(gh=30, gw=40, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(gh, gw, dim)).astype("float32")
    emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
    return Index(emb), rng


def test_cell_and_pixel_round_trip():
    idx, _ = make_index()
    for cell in (0, 17, 421, idx.gh * idx.gw - 1):
        row, col = idx.pixel_of(cell)
        assert idx.cell_of(row, col) == cell


def test_cell_of_clamps_outside_the_grid():
    idx, _ = make_index()
    assert idx.cell_of(-500, -500) == 0
    assert idx.cell_of(1e6, 1e6) == idx.gh * idx.gw - 1


def test_top_suppresses_neighbours_and_respects_exclude():
    idx, _ = make_index()
    scores = np.zeros(idx.gh * idx.gw)
    centre = 10 * idx.gw + 10
    scores[centre] = 1.0
    scores[centre + 1] = 0.9  # neighbour of the best cell: must be suppressed
    scores[20 * idx.gw + 30] = 0.8
    assert idx.top(scores, 2) == [centre, 20 * idx.gw + 30]
    assert idx.top(scores, 1, exclude={centre})[0] != centre


def test_prototype_scores_peak_at_the_example():
    idx, _ = make_index()
    cell = 5 * idx.gw + 7
    scores = idx.flat @ (idx.flat[cell] / np.linalg.norm(idx.flat[cell]))
    assert int(np.argmax(scores)) == cell


def test_classifier_scores_rank_planted_structures_first():
    idx, rng = make_index(gh=40, gw=40, dim=8)
    direction = np.zeros(8)
    direction[0] = 1.0
    planted = rng.choice(idx.gh * idx.gw, 25, replace=False)
    idx.flat[planted] += 4 * direction  # rare structures share a direction
    scores = idx.classifier_scores(list(planted[:5]), [], rng)
    ranked = np.argsort(-scores)[:25]
    assert len(set(ranked) & set(planted)) >= 18


def test_hillshade_png_is_not_blank_when_terrain_has_nodata():
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lidarprospector.terrain import hillshade

    y, x = np.mgrid[:120, :120].astype(float)
    dtm = 50 + 5 * np.sin(x / 9) * np.cos(y / 7)
    dtm[:, 80:] = np.nan  # a survey edge
    buf = io.BytesIO()
    plt.imsave(buf, hillshade(dtm), cmap="gray", vmin=0.0, vmax=1.0, format="png")
    rgba = plt.imread(io.BytesIO(buf.getvalue()))
    assert (rgba[:, :80, 3] == 1).all()  # terrain is opaque
    assert (rgba[:, 90:, 3] == 0).all()  # nodata is transparent
