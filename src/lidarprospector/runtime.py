"""Detect the machine and turn it into a processing plan.

The plan sizes tiles and worker counts from *currently available* RAM (not installed RAM): other apps
share the machine, and an over-eager planner can push it into swap or a kernel panic. A MemoryGate
re-checks free memory before each tile and waits for it if something else grabbed RAM meanwhile.
`python -m lidarprospector.runtime` prints the plan ("doctor").
"""

import contextlib
import os
import time
from dataclasses import dataclass

import psutil

# Peak bytes per pixel of a tile, measured on a 9 MP tile (see profile notes in the README).
BYTES_PER_PX = {"cpu": 370, "mps": 300, "cuda": 300}
CORE_CHOICES = (512, 768, 1024, 1536, 2048, 3072, 4096)
GB = 1 << 30


@dataclass(frozen=True)
class Hardware:
    cores: int
    ram_total: int
    ram_available: int
    device: str
    device_name: str
    device_cap: int  # bytes the accelerator will let us allocate (0 = unlimited / host RAM)


def detect_device(prefer: str = "auto") -> tuple[str, str, int]:
    try:
        import torch
    except ImportError:
        return "cpu", "CPU (install the 'embed' extra for GPU acceleration)", 0
    if prefer in ("auto", "mps") and torch.backends.mps.is_available():
        return "mps", "Apple Metal (MPS)", int(torch.mps.recommended_max_memory())
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return "cuda", props.name, int(props.total_memory)
    return "cpu", "CPU", 0


def detect(prefer: str = "auto") -> Hardware:
    device, name, cap = detect_device(prefer)
    vm = psutil.virtual_memory()
    return Hardware(os.cpu_count() or 1, vm.total, vm.available, device, name, cap)


@dataclass(frozen=True)
class Plan:
    hardware: Hardware
    mem_budget: int  # bytes this run may use at once
    tile_core: int  # pixels, side of the part of a tile we keep
    halo: int  # pixels of context around the core
    compute_workers: int
    io_workers: int
    bytes_per_px: int

    @property
    def tile_total(self) -> int:
        return self.tile_core + 2 * self.halo

    def describe(self) -> str:
        h = self.hardware
        return (
            f"device        {h.device_name}\n"
            f"cores         {h.cores}\n"
            f"RAM           {h.ram_available / GB:.1f} GB available of {h.ram_total / GB:.1f} GB\n"
            f"memory budget {self.mem_budget / GB:.1f} GB (peak, all workers)\n"
            f"tile          {self.tile_core} px core + {self.halo} px halo = {self.tile_total} px square "
            f"({self.tile_total**2 * self.bytes_per_px / GB:.1f} GB per tile in flight)\n"
            f"workers       {self.compute_workers} compute, {self.io_workers} download"
        )


def make_plan(
    halo_px: int,
    ram_fraction: float = 0.5,
    reserve_gb: float = 4.0,
    max_budget_gb: float | None = None,
    io_workers: int = 4,
    prefer_device: str = "auto",
    hardware: Hardware | None = None,
) -> Plan:
    """Largest tiles that fit: budget = fraction of (available - reserve), capped by the accelerator
    and an optional user ceiling. More compute workers only when tiles are small enough to afford them."""
    hw = hardware or detect(prefer_device)
    budget = max(0.0, hw.ram_available - reserve_gb * GB) * ram_fraction
    if hw.device_cap:
        budget = min(budget, hw.device_cap)
    if max_budget_gb is not None:
        budget = min(budget, max_budget_gb * GB)
    bpp = BYTES_PER_PX[hw.device]
    # accelerators serialise compute; on CPU, extra workers only help if memory allows
    max_workers = 1 if hw.device != "cpu" else max(1, hw.cores // 2)
    best = None
    for workers in range(max_workers, 0, -1):
        for core in CORE_CHOICES:
            if (core + 2 * halo_px) ** 2 * bpp * workers <= budget:
                best = (workers, core)
        if best:
            break
    if best is None:
        raise MemoryError(
            f"{budget / GB:.1f} GB budget is too small for even a {CORE_CHOICES[0]} px tile with "
            f"{halo_px} px halo; free some RAM or lower the filter size"
        )
    workers, core = best
    return Plan(hw, int(budget), core, halo_px, workers, io_workers, bpp)


class MemoryGate:
    """Admit work only while the machine really has the memory free; wait if something else took it."""

    def __init__(self, reserve_gb: float = 4.0, poll_s: float = 2.0, timeout_s: float = 600.0):
        self.reserve = int(reserve_gb * GB)
        self.poll_s, self.timeout_s = poll_s, timeout_s

    @contextlib.contextmanager
    def admit(self, nbytes: int):
        waited = 0.0
        while psutil.virtual_memory().available < nbytes + self.reserve:
            if waited >= self.timeout_s:
                raise MemoryError(
                    f"needed {nbytes / GB:.1f} GB free; machine stayed short for {waited:.0f} s"
                )
            time.sleep(self.poll_s)
            waited += self.poll_s
        yield


def main() -> None:
    from .tiling import required_halo

    print(make_plan(halo_px=required_halo((1,))).describe())


if __name__ == "__main__":
    main()
