import pytest

from lidarprospector.labels import classify


@pytest.mark.parametrize(
    ("name", "area_ha", "expected"),
    [
        ("Bowl barrow within Luxenborough Plantation", 0.04, "barrow"),
        ("Bell barrow 100m east of Stonehenge immediately south of the A344", 0.15, "barrow"),
        # large composite polygons are classed by structure, not by the barrows they mention
        ("The Cursus, two round barrows situated within its western end", 40.0, "linear"),
        ("Stonehenge, the Avenue, and three barrows adjacent to the Avenue", 11.8, "linear"),
        ("Henge monument 400m south of Stonehenge Cottages", 0.3, "enclosure"),
        # "linear round barrow cemetery" describes barrows, not a linear earthwork
        ("Bowl barrow forming part of a linear round barrow cemetery", 0.11, "barrow"),
        ("Long barrow and 18 round barrows forming Normanton Down round barrow cemetery", 4.7, "barrow"),
        ("Medieval fishpond", 1.0, "other"),
        (None, 1.0, "other"),
    ],
)
def test_classify(name, area_ha, expected):
    assert classify(name, area_ha) == expected
