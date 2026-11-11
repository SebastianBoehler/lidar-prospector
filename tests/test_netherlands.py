import numpy as np
import pytest

from lidarprospector.labels_osm import point_labels
from lidarprospector.sources.ahn import NATIVE, NetherlandsAHN, request_params


def test_one_metre_grid_requests_server_side_half_scaling():
    p = request_params((153000, 450000, 156000, 453000), 1.0)
    assert p["scalefactor"] == "0.5" and p["coverageId"] == "dtm_05m"
    assert p["subset"] == ["x(153000,156000)", "y(450000,453000)"]
    assert request_params((0, 0, 10, 10), NATIVE)["scalefactor"] == "1"


def test_finer_than_native_is_rejected():
    with pytest.raises(ValueError, match="0.5 m"):
        request_params((0, 0, 10, 10), 0.25)


def test_registry_style_coverage():
    ahn = NetherlandsAHN()
    assert ahn.covers(5.4, 52.07) and not ahn.covers(-1.8, 51.2)
    assert ahn.epsg == 28992 and ahn.licence == "CC0 1.0"


def test_point_labels_give_each_point_a_disc_and_a_unique_id():
    xy = np.array([[110.0, 190.0], [150.0, 150.0]])  # x, y in metres; grid origin (100, 200), 1 m cells
    near = point_labels((100, 100), 100.0, 200.0, 1.0, xy, tol_m=5.0)
    assert near[10, 10] == 1 and near[50, 50] == 2  # rows count south from y1
    assert near[10, 14] == 1 and near[10, 17] == 0  # within / beyond the 5 m tolerance
    assert set(np.unique(near)) == {0, 1, 2}
