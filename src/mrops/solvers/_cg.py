"""Conjugate Gradient replacement."""

__all__ = ["ConjugateGradient"]

import gc
from typing import Callable

from numpy.typing import NDArray

from mrinufft._array_compat import CUPY_AVAILABLE
from mrinufft._array_compat import get_array_module
from mrinufft._array_compat import with_numpy_cupy

from scipy.sparse.linalg import cg as scipy_cg

if CUPY_AVAILABLE:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import cg as cupy_cg

from .._sigpy.app import App
from .._sigpy.alg import Alg
from .._sigpy.linop import Linop

from ._monitor import Monitor
from ._reginversion import build_extended_square_system


class ConjugateGradient(App):
    r"""
    Conjugate gradient method.

    Solves the linear system:

    .. math:: (A + \sum_{r} \lambda_{r} R_{r}^{H} R_{r}) x = b + \sum_{r} \lambda_{r} R_{r}^{H} x_{r}

    where A is a Hermitian linear operator.

    Parameters
    ----------
    A : Linop
        Linear operator or function that computes the action of A on a vector.
    b : NDArray[complex]
        Right-hand side observation vector.
    x : NDArray[complex] | None, optional
        Initial guess for the solution.
    R: Linop | list[Linop] | None, optional
        Linear operator for L2 regularization. If not specified, and ``damp != 0.0``,
        this is the identity (Tikhonov regularization).
    damp: float[float] | tuple[float] | None, optional
        Regularization strength. If scalar, assume same regularization
        for all the priors. The default is ``0.0``.
    bias: NDArray[complex] | None, optional
        Bias for L2 regularization (prior image).
        The default is ``None``.
    max_iter : int, optional
        Maximum number of iterations (default is ``10``).
    tol : float, optional
        Tolerance for stopping condition (default is ``0.0``).
    verbose : bool, optional
        Toggle whether show progress (default is ``False``).
    record_stats: bool, optional
        Toggle cost function monitoring. The default is ``False``.
    record_time : bool, optional
        Toggle wheter record runtime (default is ``False``).
    solution : NDArray[complex] | None, optional
        Ground Truth solution (to check performance). The default is ``None``.

    """

    def __init__(
        self,
        A: Linop,
        b: NDArray[complex],
        x: NDArray[complex] | None = None,
        R: Linop | list[Linop] | None = None,
        damp: float[float] | tuple[float] | None = None,
        bias: NDArray[complex] | None = None,
        max_iter: int = 10,
        tol: float = 0.0,
        verbose: bool = False,
        record_stats: bool = False,
        record_time: bool = False,
        solution: NDArray[complex] | None = None,
    ):
        _alg = _ConjugateGradient(
            A, b, x, R, damp, bias, max_iter, tol, record_stats, verbose, solution
        )
        super().__init__(_alg, False, False, record_time)

    def _output(self):
        gc.collect()
        if CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()
        return self.alg.x


class _ConjugateGradient(Alg):
    def __init__(
        self,
        A: Linop,
        b: NDArray[complex],
        x: NDArray[complex],
        R: Linop | list[Linop] | None = None,
        damp: float[float] | tuple[float] | None = None,
        bias: NDArray[complex] | None = None,
        max_iter: int = 10,
        tol: float = 0.0,
        record_stats: bool = False,
        verbose: bool = True,
        solution: NDArray[complex] | None = None,
    ):
        A_reg, b_reg = build_extended_square_system(A, b, R, damp, bias)
        self.A = A_reg
        self.b = b_reg
        self.x = x
        self.tol = tol
        self._finished = False
        self._verbose = verbose
        self._record_stats = record_stats
        self._solution = solution

        super().__init__(max_iter)

    def update(self):  # noqa
        if self.x is None:
            self.x = 0 * self.b

        # get shape
        shape = self.b.shape

        # build callable
        if self._record_stats:
            callback = Monitor(self.A_reg, self.b_reg, self._verbose, self._solution)
        else:
            callback = None

        # actual run
        self.x = _cg(
            self.A,
            self.b.ravel(),
            self.x.ravel(),
            tol=self.tol,
            maxiter=self.max_iter,
            callback=callback,
        )  # here we let scipy/cupy handle steps.

        # reshape back
        self.b = self.b.reshape(*shape)
        self.x = self.x.reshape(*shape)

        self._finished = True

    def _done(self):
        return self._finished


@with_numpy_cupy
def _cg(A, b, x0, *, tol=0, maxiter=None, callback=None):
    if maxiter is not None and maxiter > 0:
        if get_array_module(b).__name__ == "numpy":
            return scipy_cg(
                A, b, x0, atol=tol, rtol=tol, maxiter=maxiter, callback=callback
            )[0]
        else:
            return cupy_cg(A, b, x0, atol=tol, maxiter=maxiter, callback=callback)[0]
    return x0
