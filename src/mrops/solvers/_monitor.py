"""Callback for optimizers."""

__all__ = ["Monitor"]

from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from dataclasses import dataclass

from mrinufft._array_compat import get_array_module


@dataclass
class Monitor:
    A_reg: LinearOperator | NDArray
    b_reg: NDArray
    verbose: bool = False
    solution: NDArray | None = None
    cost: list[float] = None
    _iter: int = 0
    _error: list[float] = None

    def __post_init__(self):
        self.cost = []
        self._error = []

    def __call__(self, input):
        xp = get_array_module(input)
        residual = self.A_reg @ input - self.b_reg
        self.cost.append(0.5 * xp.linalg.norm(residual) ** 2)  # Least squares cost
        if self.verbose and self.solution is not None:
            self._error.append(xp.linalg.norm(input - self.solution))
            print(
                f"Iteration: {self._iter} | Cost: {self.cost[-1]:.6e} | Error: {self._error[-1]:.6e}"
            )
        elif self.verbose:
            print(f"Iteration: {self._iter} | Cost: {self.cost[-1]:.6e}")
        self._iter += 1
