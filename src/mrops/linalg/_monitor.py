"""Callback for optimizers."""

__all__ = ["Monitor"]

import time

from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from dataclasses import dataclass

from mrinufft._array_compat import get_array_module, CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


@dataclass
class Monitor:
    """Utility class to monitor solver and record time."""

    A_reg: LinearOperator | NDArray
    b_reg: NDArray
    verbose: bool = False
    solution: NDArray | None = None
    _cost: list[float] = None
    _iter: int = 0
    _error: list[float] = None
    _start_cpu: float = 0.0
    _start_gpu: float = 0.0
    _stop_cpu: float = 0.0
    _stop_gpu: float = 0.0
    _time_cpu: float | None = None
    _time_gpu: float | None = None

    def __post_init__(self):
        self._cost = []
        self._error = []

    def __call__(self, input):
        xp = get_array_module(input)
        residual = self.A_reg @ input - self.b_reg
        self._cost.append(0.5 * xp.linalg.norm(residual) ** 2)  # Least squares cost
        if self.verbose and self.solution is not None:
            self._error.append(xp.linalg.norm(input - self.solution))
            print(
                f"Iteration: {self._iter} | Cost: {self._cost[-1]:.6e} | Error: {self._error[-1]:.6e}"
            )
        elif self.verbose:
            print(f"Iteration: {self._iter} | Cost: {self._cost[-1]:.6e}")
        self._iter += 1

    def start_timer(self):
        if CUPY_AVAILABLE:
            self._start_gpu = cp.cuda.Event()
            self._start_gpu.record()
        self._start_cpu = time.perf_counter()

    def stop_timer(self):
        self._stop_cpu = time.perf_counter()
        if CUPY_AVAILABLE:
            self._stop_gpu.record()
            self._stop_gpu.synchronize()
            self._time_gpu = cp.cuda.get_elapsed_time(self._start_gpu, self._stop_gpu)
        self._time_cpu = self._stop_cpu - self._start_cpu

    @property
    def time(self):
        if self._time_cpu and self._time_gpu:
            return (self._time_cpu, self._time_gpu)
        elif self._time_cpu:
            return self._time_cpu

    @property
    def history(self):
        if self._error is not None:
            return (self._cost, self._error)
        else:
            return self._cost
