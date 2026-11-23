"""Real PyTorch refuses memory-stat calls on a CUDA device before CUDA is initialised:
    RuntimeError: Invalid device argument
The smoke run only worked because reading the GPU names happened to initialise CUDA first, and the
probe, which did not, failed on the Kaggle run. These tests use a fake torch that behaves the same.
"""

import sys
import types

import pytest

from ptlegal_eval import models


class FakeCuda:
    def __init__(self, devices: int = 2) -> None:
        self.initialised = False
        self.devices = devices
        self.calls: list[tuple[str, int]] = []

    def is_available(self) -> bool:
        return self.devices > 0

    def init(self) -> None:
        self.initialised = True

    def device_count(self) -> int:
        return self.devices

    def _need_init(self) -> None:
        if not self.initialised:
            raise RuntimeError("Invalid device argument")

    def reset_peak_memory_stats(self, i: int) -> None:
        self._need_init()
        self.calls.append(("reset", i))

    def max_memory_reserved(self, i: int) -> int:
        self._need_init()
        return 2**30 * 9

    def memory_allocated(self, i: int) -> int:
        self._need_init()
        return 0

    def synchronize(self) -> None:
        self._need_init()

    def empty_cache(self) -> None:
        pass


@pytest.fixture
def fake_torch(monkeypatch):
    cuda = FakeCuda()
    module = types.SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", module)
    return cuda


def test_resetting_peaks_initialises_cuda_first(fake_torch) -> None:
    models.reset_gpu_peaks()  # raised "Invalid device argument" before the fix
    assert fake_torch.initialised and fake_torch.calls == [("reset", 0), ("reset", 1)]


def test_reading_peaks_works_in_a_fresh_process(fake_torch) -> None:
    assert models.gpu_peaks_gib() == {"cuda:0": 9.0, "cuda:1": 9.0}


def test_no_gpu_is_not_an_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=FakeCuda(devices=0)))
    models.reset_gpu_peaks()
    assert models.gpu_peaks_gib() == {}
