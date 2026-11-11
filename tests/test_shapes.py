import numpy as np

from lidarprospector.shapes import depressions, linear_features

from .test_terrain import bump


def test_closed_pit_is_a_depression():
    found = depressions(-bump(200, height=0.8, sigma=4.0))
    assert len(found) == 1
    assert abs(found[0].row - 100) < 3
    assert abs(found[0].col - 100) < 3


def test_valley_side_is_not_a_depression():
    y = np.mgrid[:200, :200][0].astype(float)
    assert depressions(0.02 * y) == []


def test_long_straight_bank_is_found_and_round_mound_is_not():
    x = np.mgrid[:300, :300][1].astype(float)
    noise = np.random.default_rng(0).normal(0, 0.005, x.shape)  # real terrain is never noiseless
    bank = 0.5 * np.exp(-((x - 150) ** 2) / (2 * 3.0**2)) + noise
    found = linear_features(bank, border=10)
    assert any(line.kind == "bank" and line.length_m > 200 for line in found)
    assert linear_features(bump(300, height=1.0, sigma=6.0), border=10) == []
