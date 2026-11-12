import pytest

pytest.importorskip("torch")

from lidarprospector.weak_supervision import is_benchmark_type, label_maps  # noqa: E402

BBOX = (1000, 2000, 1400, 2400)


def mon(name, x, y, half=10, area=0.04):
    ring = [
        [x - half, y - half],
        [x + half, y - half],
        [x + half, y + half],
        [x - half, y + half],
        [x - half, y - half],
    ]
    return {"name": name, "area_ha": area, "rings": [ring]}


def test_benchmark_type_names():
    assert is_benchmark_type(mon("Bowl barrow", 0, 0))
    assert is_benchmark_type(mon("Round cairn on the moor", 0, 0))
    assert is_benchmark_type(mon("Hut circle settlement", 0, 0))
    assert not is_benchmark_type(mon("Iron Age hillfort", 0, 0, area=3.0))


def test_typed_mode_marks_barrows_and_ignores_the_rest():
    mons = [mon("Round barrow", 1100, 2300), mon("Hillfort", 1300, 2100, area=3.0)]
    lab = label_maps(BBOX, mons, "typed", shape=(400, 400))
    pos, ign = lab[0] > 0, lab[1] > 0
    assert pos[100, 100]  # row = top - Y = 2400 - 2300, col = X - left
    assert not pos[300, 300] and ign[300, 300]
    assert not (pos & ign).any()


def test_other_mode_is_the_complement_in_what_it_calls_positive():
    mons = [mon("Round barrow", 1100, 2300), mon("Hillfort", 1300, 2100, area=3.0)]
    lab = label_maps(BBOX, mons, "other", shape=(400, 400))
    pos, ign = lab[0] > 0, lab[1] > 0
    assert pos[300, 300] and not pos[100, 100]
    assert ign[100, 100]  # the benchmark type is ignored, not taught as background


def test_a_barrow_cemetery_is_not_a_positive():
    big = mon("Round barrow cemetery", 1200, 2200, half=60, area=1.4)
    lab = label_maps(BBOX, [big, mon("Round barrow", 1100, 2300)], "typed", shape=(400, 400))
    pos, ign = lab[0] > 0, lab[1] > 0
    assert pos[100, 100] and not pos[200, 200]  # the single barrow, not the cemetery polygon
    assert ign[200, 200]
