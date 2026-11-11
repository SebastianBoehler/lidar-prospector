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
        ("Round cairn on Danby Rigg", 0.05, "barrow"),
        ("Medieval fishpond", 1.0, "other"),
        (None, 1.0, "other"),
        # some records have no recorded area: never a tight single barrow, but not a crash
        ("Bowl barrow on the moor", None, "barrow"),
    ],
)
def test_classify(name, area_ha, expected):
    assert classify(name, area_ha) == expected


def test_tight_barrow_helper_handles_missing_areas():
    from lidarprospector.labels import is_tight_barrow

    assert is_tight_barrow({"cls": "barrow", "area_ha": 0.1})
    assert not is_tight_barrow({"cls": "barrow", "area_ha": 3.0})
    assert not is_tight_barrow({"cls": "barrow", "area_ha": None})
    assert not is_tight_barrow({"cls": "linear", "area_ha": 0.1})
