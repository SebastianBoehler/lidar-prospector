from lidarprospector.labels_scotland import accuracy_m, site_type


def test_site_types():
    assert site_type("HUT CIRCLE (PREHISTORIC)") == "hut circle"
    assert site_type("CAIRNFIELD (PERIOD UNASSIGNED), HUT CIRCLE (PREHISTORIC)") == "hut circle"
    assert site_type("CAIRN (PREHISTORIC)") == "cairn"
    assert site_type("BURIAL CAIRN (PREHISTORIC)") == "cairn"
    assert site_type("BARROW (PREHISTORIC)") == "round barrow"
    assert site_type("STANDING STONE (PREHISTORIC)") == "stone setting"


def test_excluded_types():
    assert site_type("CLEARANCE CAIRN (PERIOD UNASSIGNED)") is None
    assert site_type("CHAMBERED CAIRN (NEOLITHIC)") is None
    assert site_type("CAIRNFIELD (PREHISTORIC)") is None
    assert (
        site_type("CAIRN (PERIOD UNASSIGNED)(POSSIBLE)") is None
    )  # possible records are not training labels
    assert site_type("LONG BARROW (NEOLITHIC)") is None
    assert site_type(None) is None


def test_accuracy_text():
    assert accuracy_m("NGR given to the nearest 1m") == 1
    assert accuracy_m("NGR given to the nearest 10m") == 10
    assert accuracy_m("NGR given to the nearest 1km") == 1000
    assert accuracy_m("General location") == 5000
