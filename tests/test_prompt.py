import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lidarprospector.promptable import fit_probe, relief_band, template_scores  # noqa: E402


def ring(n=64, r0=9, r1=13, h=0.5):
    y, x = np.mgrid[:n, :n]
    d = np.hypot(y - n // 2, x - n // 2)
    return h * ((d >= r0) & (d <= r1)).astype("float32")


def scene(seed=0):
    rng = np.random.default_rng(seed)
    z = 0.02 * rng.standard_normal((400, 400)).astype("float32")
    z[60:124, 60:124] += ring()  # ring A
    z[250:314, 210:274] += ring()  # ring B: same shape elsewhere
    y, x = np.mgrid[:400, :400]
    z += 0.5 * np.exp(-((y - 330) ** 2 + (x - 90) ** 2) / (2 * 6.0**2)).astype("float32")  # a dome decoy
    return z


def test_template_prompt_finds_the_same_shape_and_prefers_it_to_a_decoy():
    z = scene()
    rel = relief_band(z)
    patch = rel[92 - 15 : 92 + 16, 92 - 15 : 92 + 16][None]  # 31 x 31 around ring A's centre (92, 92)
    s = template_scores(rel, patch, "cpu", pool=1).numpy()
    ring_b = s[282 - 8 : 282 + 9, 242 - 8 : 242 + 9].max()
    dome = s[330 - 8 : 330 + 9, 90 - 8 : 90 + 9].max()
    flat = np.median(s)
    assert ring_b > 0.5 and ring_b > dome and ring_b > flat + 0.3


def test_template_matching_is_rotation_invariant_for_quarter_turns():
    z = scene()
    rel = relief_band(z)
    patch = rel[92 - 15 : 92 + 16, 92 - 15 : 92 + 16]
    a = template_scores(rel, patch[None], "cpu", pool=1).numpy()
    b = template_scores(rel, np.rot90(patch)[None].copy(), "cpu", pool=1).numpy()
    assert np.allclose(a, b, atol=1e-4)


def test_probe_separates_examples_from_background_with_a_handful_of_examples():
    rng = np.random.default_rng(1)
    bg = rng.normal(0, 1, (2000, 8))
    pos = rng.normal(0, 1, (9, 8)) + np.array([3, 3, 0, 0, 0, 0, 0, 0])
    w, b = fit_probe(pos, bg)
    test_pos = rng.normal(0, 1, (200, 8)) + np.array([3, 3, 0, 0, 0, 0, 0, 0])
    test_bg = rng.normal(0, 1, (200, 8))
    assert (test_pos @ w + b).mean() > (test_bg @ w + b).mean() + 1.0


def test_prompt_search_returns_other_instances_of_the_clicked_shape():
    from lidarprospector.promptable import Extractor, PromptSearch

    z = scene()

    def features(dtm):  # a tiny stand-in extractor: relief at two scales
        rel = relief_band(dtm, 4.0)
        return np.stack([rel, relief_band(dtm, 10.0)], axis=-1)

    ps = PromptSearch(Extractor("relief", features, 2), z, "cpu")
    hits = ps.search([(92, 92)], k=5)  # click ring A's centre
    near_b = [h for h in hits if abs(h[0] - 282) < 14 and abs(h[1] - 242) < 14]
    assert near_b, f"ring B not among candidates: {hits}"
    assert all(np.hypot(h[0] - 92, h[1] - 92) > 8 for h in hits)  # the clicked place itself is excluded
