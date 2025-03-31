"""Conjugate Gradient replacement."""

__all__ = ["cg", "ConjugateGradient"]

import gc

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


def cg(
    A: Linop,
    b: NDArray[complex],
    damp: float | list[float] | tuple[float] = 0.0,
    R: Linop | list[Linop] | None = None,
    bias: NDArray[complex] | None = None,
    x: NDArray[complex] | None = None,
    max_iter: int = 10,
    tol: float = 0.0,
    verbose: bool = False,
    record_stats: bool = False,
    record_time: bool = False,
    solution: NDArray[complex] | None = None,
):
    r"""
    Conjugate gradient method.

    Solves the regularized linear problem::

        minimize 0.5 * || A @ x - b ||^2_2 + 0.5 * Σ_i damp_i * || R_i @ x - bias_i ||^2_2

    where:
        - ``A`` is the normal (Hermitian) linear operator.
        - ``b`` are the measured data.
        - Each ``R_i`` is a regularization linear operator.
        - Each ``damp_i`` is a scalar controlling the strength of ``R_i``.
        - Each ``bias_i`` is a prior for L2 regularization.

    Notes
    -----
    Setting ``damp`` to ``0.0`` implies solving unregularized problem, regardless
    of provided regularizers and biases. Setting ``damp`` to a non-zero scalar
    without providing regularizers implies standard L2 (Tikhonov) regularization
    (i.g., ``R = I``; ``bias = 0.0``).

    Parameters
    ----------
    A : Linop
        Linear operator or function that computes the action of A on a vector.
    b : NDArray[complex]
        Right-hand side observation vector.
    damp: float | list[float] | tuple[float], optional
        Regularization strength. If scalar, assume same regularization
        for all the priors. The default is ``0.0``.
    R: Linop | list[Linop] | None, optional
        Linear operator for L2 regularization. If not specified, and ``damp != 0.0``,
        this is the identity (Tikhonov regularization).
    bias: NDArray[complex] | None, optional
        Bias for L2 regularization (prior image).
        The default is ``None``.
    x : NDArray[complex] | None, optional
        Initial guess for the solution.
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

    Returns
    -------
    NDArray[complex]
        Solution to the problem.

    """
    solver = ConjugateGradient(
        A,
        b,
        damp,
        R,
        bias,
        x,
        max_iter,
        tol,
        verbose,
        record_stats,
        record_time,
        solution,
    )
    return solver.run()


class ConjugateGradient(App):
    r"""
    Conjugate gradient method.

    Solves the regularized linear problem::

        minimize 0.5 * || A @ x - b ||^2_2 + 0.5 * Σ_i damp_i * || R_i @ x - bias_i ||^2_2

    where:
        - ``A`` is the normal (Hermitian) linear operator.
        - ``b`` are the measured data.
        - Each ``R_i`` is a regularization linear operator.
        - Each ``damp_i`` is a scalar controlling the strength of ``R_i``.
        - Each ``bias_i`` is a prior for L2 regularization.

    Parameters
    ----------
    A : Linop
        Linear operator or function that computes the action of A on a vector.
    b : NDArray[complex]
        Right-hand side observation vector.
    damp: float | list[float] | tuple[float], optional
        Regularization strength. If scalar, assume same regularization
        for all the priors. The default is ``0.0``.
    R: Linop | list[Linop] | None, optional
        Linear operator for L2 regularization. If not specified, and ``damp != 0.0``,
        this is the identity (Tikhonov regularization).
    bias: NDArray[complex] | None, optional
        Bias for L2 regularization (prior image).
        The default is ``None``.
    x : NDArray[complex] | None, optional
        Initial guess for the solution.
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
        damp: float | list[float] | tuple[float] = 0.0,
        R: Linop | list[Linop] | None = None,
        bias: NDArray[complex] | None = None,
        x: NDArray[complex] | None = None,
        max_iter: int = 10,
        tol: float = 0.0,
        verbose: bool = False,
        record_stats: bool = False,
        record_time: bool = False,
        solution: NDArray[complex] | None = None,
    ):
        _alg = _ConjugateGradient(
            A,
            b,
            damp,
            R,
            bias,
            x,
            max_iter,
            tol,
            verbose,
            record_stats,
            record_time,
            solution,
        )
        super().__init__(_alg, False, False, False)

    def _output(self):
        gc.collect()
        if CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()
        return self.alg.x


# %% utils
class _ConjugateGradient(Alg):
    def __init__(
        self,
        A: Linop,
        b: NDArray[complex],
        damp: float | list[float] | tuple[float] | None = None,
        R: Linop | list[Linop] | None = None,
        bias: NDArray[complex] | None = None,
        x: NDArray[complex] | None = None,
        max_iter: int = 10,
        tol: float = 0.0,
        verbose: bool = False,
        record_stats: bool = False,
        record_time: bool = False,
        solution: NDArray[complex] | None = None,
    ):
        A_reg, b_reg = build_extended_square_system(A, b, damp, R, bias)
        self.ishape = A.ishape
        self.oshape = A.oshape
        self.A = A_reg
        self.b = b_reg
        self.x = x
        self.tol = tol
        self._finished = False
        self._verbose = verbose
        self._record_stats = record_stats
        self._record_time = record_time
        self._solution = solution

        super().__init__(max_iter)

    def update(self):  # noqa
        if self.x is None:
            self.x = 0 * self.b

        # build callable
        if self._record_stats:
            callback = Monitor(self.A, self.b, self._verbose, self._solution)
            timer = callback
        elif self._record_time:
            callback = None
            timer = Monitor(self.A, self.b, self._verbose, self._solution)
        else:
            callback = None

        # start timer
        if self._record_time:
            timer.start_timer()

        # actual run
        if self._verbose:
            print("CG start")
        self.x = _cg(
            self.A,
            self.b.ravel(),
            self.x.ravel(),
            tol=self.tol,
            maxiter=self.max_iter,
            callback=callback,
        )  # here we let scipy/cupy handle steps.
        self.x = self.x.reshape(*self.ishape)
        if self._verbose:
            print("CG end")

        # stop timer
        if self._record_time:
            timer.stop_timer()
            self.time = timer.time
            if self._verbose:
                print(f"Elapsed time: {self.time} s")
        if self._record_stats:
            self.history = callback.history
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
