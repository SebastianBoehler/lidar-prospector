import pytest

from lidarprospector.runtime import GB, Hardware, MemoryGate, make_plan


def hw(avail_gb, device="cpu", cores=8, cap_gb=0):
    return Hardware(cores, 64 * GB, int(avail_gb * GB), device, device, int(cap_gb * GB))


def test_more_free_ram_gives_bigger_tiles():
    small = make_plan(160, hardware=hw(8))
    big = make_plan(160, hardware=hw(48))
    assert big.tile_core > small.tile_core
    assert big.mem_budget > small.mem_budget


def test_budget_never_exceeds_free_memory_after_reserve():
    plan = make_plan(160, hardware=hw(20), ram_fraction=0.5, reserve_gb=4)
    assert plan.mem_budget <= (20 - 4) * 0.5 * GB
    assert plan.tile_total**2 * plan.bytes_per_px * plan.compute_workers <= plan.mem_budget


def test_accelerator_cap_and_user_ceiling_both_bind():
    assert make_plan(160, hardware=hw(48, "mps", cap_gb=3)).mem_budget <= 3 * GB
    assert make_plan(160, hardware=hw(48), max_budget_gb=2).mem_budget <= 2 * GB


def test_accelerators_use_one_compute_worker_but_cpu_may_use_more():
    assert make_plan(160, hardware=hw(48, "mps")).compute_workers == 1
    assert make_plan(160, hardware=hw(60, "cpu", cores=16)).compute_workers > 1


def test_too_little_memory_fails_loudly():
    with pytest.raises(MemoryError, match="too small"):
        make_plan(160, hardware=hw(4.2))


def test_gate_times_out_instead_of_hanging():
    gate = MemoryGate(reserve_gb=0, poll_s=0.01, timeout_s=0.05)
    with pytest.raises(MemoryError), gate.admit(10**15):
        pass
    with MemoryGate(reserve_gb=0).admit(1):  # trivially available
        pass
