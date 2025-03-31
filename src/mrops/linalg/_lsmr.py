"""LSMR solver."""

__all__ = ["lsmr", "LSMR"]

import gc
import warnings

from numpy.typing import NDArray

from mrinufft._array_compat import CUPY_AVAILABLE
from mrinufft._array_compat import get_array_module
from mrinufft._array_compat import with_numpy_cupy

from scipy.sparse.linalg import lsmr as scipy_lsmr

if CUPY_AVAILABLE:
    import cupy as cp
    from ._cupy_lsmr import lsmr as cupy_lsmr

from .._sigpy.app import App
from .._sigpy.alg import Alg
from .._sigpy.linop import Linop

from ._monitor import Monitor
from ._reginversion import build_extended_system


def lsmr(
    A: Linop,
    b: NDArray[complex],
    damp: float | list[float] | tuple[float] = 0.0,
    R: Linop | list[Linop] | None = None,
    x: NDArray[complex] | None = None,
    bias: NDArray[complex] | None = None,
    max_iter: int = 10,
    tol: float = 0.0,
    verbose: bool = False,
    record_time: bool = False,
) -> NDArray[complex]:
    r"""
    LSMR method.

    Solves the regularized linear problem::

        minimize || [ b ] - [ A          ] x                                ||_2
                 || [ sqrt(damp_1) R_1.H @ bias_1 ]    [ sqrt(damp_1) R_1 ] ||
                 || [ sqrt(damp_2) R_2.H @ bias_2 ]    [ sqrt(damp_2) R_2 ] ||
                 ||   ...     ...                                           ||
                 || [ sqrt(damp_r) R_r.H @ bias_r ]    [ sqrt(damp_r) R_r ] ||

    where:
        - ``A`` is the forward linear operator.
        - ``b`` are the measured data.
        - Each ``R_i`` is a regularization linear operator.
        - Each ``damp_i`` is a scalar controlling the strength of ``R_i``.
        - Each ``bias_i`` is a prior for L2 regularization.

    Notes
    -----
    Setting ``damp`` to ``0.0`` implies solving unregularized problem, regardless
    of provided regularizers and biases. Setting ``damp`` to a non-zero scalar
    without providing regularizers implies standard L2 (Tikhonov) regularization
    (i.g., ``R = I``; ``bias = 0.0``). In this case, please provide ``damp**2``
    if you want to mimick SciPy/CuPy lsmr behaviour, as ``damp`` is internally
    rescaled to ``damp**0.5`` here.

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
    x : NDArray[complex]
        Initial guess for the solution.
    max_iter : int, optional
        Maximum number of iterations (default is ``10``).
    tol : float, optional
        Tolerance for stopping condition (default is ``0.0``).
    verbose : bool, optional
        Toggle whether show progress (default is ``False``).
    record_time : bool, optional
        Toggle wheter record runtime (default is ``False``).

    Returns
    -------
    NDArray[complex]
        Solution to the problem.

    """
    solver = LSMR(A, b, damp, R, bias, x, max_iter, tol, verbose, record_time)
    return solver.run()


class LSMR(App):
    r"""
    LSMR method.

    Solves the regularized linear problem::

        minimize || [ b ] - [ A          ] x                                ||_2
                 || [ sqrt(damp_1) R_1.H @ bias_1 ]    [ sqrt(damp_1) R_1 ] ||
                 || [ sqrt(damp_2) R_2.H @ bias_2 ]    [ sqrt(damp_2) R_2 ] ||
                 ||   ...     ...                                           ||
                 || [ sqrt(damp_r) R_r.H @ bias_r ]    [ sqrt(damp_r) R_r ] ||

    where:
        - ``A`` is the forward linear operator.
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
    x : NDArray[complex]
        Initial guess for the solution.
    max_iter : int, optional
        Maximum number of iterations (default is ``10``).
    tol : float, optional
        Tolerance for stopping condition (default is ``0.0``).
    verbose : bool, optional
        Toggle whether show progress (default is ``False``).
    record_time : bool, optional
        Toggle wheter record runtime (default is ``False``).

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
        record_time: bool = False,
    ):
        _alg = _LSMR(A, b, damp, R, bias, x, max_iter, tol, verbose, record_time)
        super().__init__(_alg, False, False, False)

    def _output(self):
        gc.collect()
        if CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()
        return self.alg.x


# %% utils
class _LSMR(Alg):
    def __init__(
        self,
        A: Linop,
        b: NDArray[complex],
        damp: float | list[float] | tuple[float] | None = None,
        R: Linop | list[Linop] | None = None,
        bias: NDArray[complex] | None = None,
        x: NDArray[complex] = None,
        max_iter: int = 10,
        tol: float = 0.0,
        verbose: bool = False,
        record_time: bool = False,
    ):
        A_reg, b_reg = build_extended_system(A, b, R, damp, bias)
        self.ishape = A.ishape
        self.oshape = A.oshape
        self.A = A_reg
        self.b = b_reg
        self.x = x
        self.tol = tol
        self._finished = False
        self._verbose = verbose
        self._record_time = record_time

        super().__init__(max_iter)

    def update(self):  # noqa
        # start timer
        if self._record_time:
            timer = Monitor(self.A, self.b)
            timer.start_timer()

        # actual run
        if self._verbose:
            print("LSMR start")
        self.x = _lsmr(
            self.A,
            self.b.ravel(),
            self.x,
            tol=self.tol,
            maxiter=self.max_iter,
        )  # here we let scipy/cupy handle steps.
        self.x = self.x.reshape(*self.ishape)
        if self._verbose:
            print("LSMR end")

        # stop timer
        if self._record_time:
            timer.stop_timer()
            self.time = timer.time
            if self._verbose:
                print(f"Elapsed time: {self.time} s")
        self._finished = True

    def _done(self):
        return self._finished


@with_numpy_cupy
def _lsmr(A, b, x0=None, tol=0, maxiter=None):
    if x0 is not None:
        x0 = x0.ravel()
    if get_array_module(b).__name__ == "numpy":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = scipy_lsmr(A, b, x0=x0, atol=tol, btol=tol, maxiter=maxiter)[0]
        return res.astype(b.dtype)
    else:
        return cupy_lsmr(A, b, x0=x0, atol=tol, btol=tol, maxiter=maxiter)[0]
