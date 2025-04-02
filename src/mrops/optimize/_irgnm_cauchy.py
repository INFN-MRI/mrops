"""Iteratively Renormalized Gauss Newton Method with Cauchy stabilization and line search solver."""

__all__ = ["IrgnmCauchy"]

import gc

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from mrinufft._array_compat import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp

from .._sigpy.app import App
from .._sigpy.alg import Alg
from .._sigpy import get_device

from ..base import NonLinop
from ._irgnm import IrgnmBase


class IrgnmCauchy(App):
    """
    Iteratively Regularized Gauss-Newton Method (IRGNM) algorithm with Cauchy
    stabilizer.

    This class accepts any nonlinear operator (implementing the NonlinearOperator
    interface) along with custom initialization and postprocessing routines.

    The method iteratively solves a linearized problem of the form

        [F'(x)^H F'(x) + α I] dx = F'(x)^H (y - F(x)) + α (x0 - x),

    using Cauchy-stabilized Gauss-Newton algorithm with linesearch, and then updates x ← x + dx.

    Parameters
    ----------
    A : NonLinop
        The nonlinear operator A.
    y : NDArray[complex | float]
        Observation.
    x : NDArray[comples | float]
        Variable.
    x0 : NDArray[comples | float]
        Regularization bias.
    alpha : float | complex, optional
        Regularization parameter. If complex, use real part to regularize
        real part of x and imaginary part to regularize imaginary part of x.
    linesearch_cost : Callable
        Linesearch cost function
    linesearch_iter : int, optional
        Number of inner (Linesearch) iterations (default is ``5``).
    max_iter : int, optional
        Number of outer (Gauss-Newton) iterations (default is ``10``).
    weights: NDArray[float] | None, optional
        Weights for Cauchy stabilization. It has the same shape as ``x`` (local stabilization).
        The default is ``1.0``-
    show_pbar : bool, optional
        Toggle whether show progress bar (default is ``False``).
    leave_pbar : bool, optional
        Toggle whether to leave progress bar after finished (default is ``True``).
    record_time : bool, optional
        Toggle wheter record runtime (default is ``False``).

    """

    def __init__(
        self,
        A: NonLinop,
        b: NDArray[complex | float],
        x: NDArray[complex | float],
        x0: NDArray[complex | float],
        alpha: complex | float,
        linesearch_costfun: Callable,
        linesearch_iter: int = 5,
        max_iter: int = 10,
        weights: NDArray[float] | None = None,
        show_pbar: bool = False,
        leave_pbar: bool = True,
        record_time: bool = False,
    ):
        _alg = _IrgnmCauchy(
            A, b, x, x0, alpha, linesearch_costfun, linesearch_iter, max_iter, weights
        )
        super().__init__(_alg, show_pbar, leave_pbar, record_time)

    def _output(self):
        gc.collect()
        if CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()
        return self.alg.x


class _IrgnmCauchy(IrgnmBase):
    def __init__(
        self,
        A: NonLinop,
        b: NDArray[complex | float],
        x: NDArray[complex | float],
        x0: NDArray[complex | float],
        alpha: complex | float,
        linesearch_costfun: Callable,
        linesearch_iter: int = 5,
        max_iter: int = 10,
        weights: NDArray[float] | None = None,
    ):
        super().__init__(A, b, x, max_iter, 1.0, q=0.0)
        self.x0 = x0
        if np.isreal(alpha):
            self.alphaRe = alpha
            self.alphaIm = alpha
        else:
            self.alphaRe = np.real(alpha)
            self.alphaIm = np.imag(alpha)
        self.linesearch_costfun = linesearch_costfun
        self.linesearch_iter = linesearch_iter
        if weights is None:
            device = get_device(x)
            xp = device.xp
            with device:
                weights = xp.ones(x.shape, dtype=xp.float32)
        self._weighted_mean = WeightedMean(weights)

    def setup_solver(self):  # noqa
        rhs = self.A.forward() - self.b

        # Compute jacobian
        JRe, JIm = self.A.jacobian()

        # Compute gradient G = J^T * r
        bias = self.x - self.x0
        gRe = (JRe.conj() * rhs).sum(axis=0).real + self.alphaRe**2 * bias.real
        gIm = (JIm.conj() * rhs).sum(axis=0).real + self.alphaIm**2 * bias.imag
        G = gRe + 1j * gIm  # Complex gradient

        # Compute approximate Hessian
        H1 = (JRe.conj() * JRe).sum(axis=0).real + self.alphaRe**2
        H2 = (JRe.conj() * JIm).sum(axis=0).real
        H3 = (JIm.conj() * JIm).sum(axis=0).real + self.alphaIm**2

        # Compute Cauchy step size
        GG = gRe**2 + gIm**2
        GHG = gRe**2 * H1 + 2 * gRe * gIm * H2 + gIm**2 * H3

        # Stabilize step size
        damp = self._weighted_mean(GHG) / 1000
        step = GG / (GHG + damp) / self.linesearch_iter
        dx = -step * G
        self.solver = LineSearch(
            self.linesearch_costfun, self.x, dx, self.linesearch_iter
        )

    def run_solver(self):  # noqa
        return self.solver.run()  # return dx

    def _update(self):
        self.A.update(self.x)
        self.setup_solver()
        return (
            self.run_solver()
        )  # for linesearch, solver returns directly x rather than dx


# %% utils
class LineSearch(App):
    """Basic linesearch algorithm."""

    def __init__(
        self,
        costfun: Callable,
        x: NDArray[complex],
        dx: NDArray[complex],
        max_iter: int = 10,
    ):
        _alg = _LineSearch(
            costfun,
            x,
            dx,
            max_iter,
        )
        super().__init__(_alg, False, False, False)

    def _output(self):
        gc.collect()
        if CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()
        return self.alg.x


class _LineSearch(Alg):
    """Basic linesearch algorithm."""

    def __init__(
        self,
        costfun: Callable,
        x: NDArray[complex],
        dx: NDArray[complex],
        max_iter: int = 5,
    ):
        self.device = get_device(x)
        self.costfun = costfun  # cost function to be optimized
        self.x = x  # solution to be optimized
        self.dx = dx  # solution update step
        super().__init__(max_iter)

    def _update(self):
        xp = self.device.xp
        ok = self.costfun(self.x + self.dx) < self.costfun(self.x)
        not_ok = xp.logical_not(ok)
        self.x[ok] += self.dx[ok]
        self.dx[not_ok] /= 10.0

    def _done(self):
        return self.iter >= self.max_iter


# %% TODO: avoid code duplication
class WeightedMean:
    """
    Apply weighted mean

    Parameters
    ----------
    weights: NDArray[float]

    """

    def __init__(self, weights):
        self._weights = weights

    def __call__(self, input):
        return _weighted_mean(input, self._weights)


def _weighted_mean(data, weights):
    """Compute the weighted mean of an array."""
    return (data * weights).sum() / (weights).sum()
