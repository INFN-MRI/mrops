"""Tikhonov regularized lstsq solver for batched matrices."""

__all__ = ["lstsq", "LSTSQ"]

import gc

from numpy.typing import NDArray

import torch

from mrinufft._array_compat import CUPY_AVAILABLE
from mrinufft._array_compat import with_torch

if CUPY_AVAILABLE:
    import cupy as cp

from .._sigpy.app import App
from .._sigpy.alg import Alg

from ._monitor import Monitor
from ._reginversion import build_extended_system


def lstsq(
    A: NDArray[complex | float],
    b: NDArray[complex | float],
    damp: float | list[float] | tuple[float] | None = None,
    R: NDArray[complex | float] | list[NDArray[complex | float]] | None = None,
    bias: NDArray[complex | float] | None = None,
    verbose: bool = False,
    record_time: bool = False,
):
    r"""
    Regularized linear least squares for dense problems.

    Solves the regularized linear problem::

        minimize 0.5 * || A @ x - b ||^2_2 + 0.5 * Σ_i damp_i * || R_i @ x - bias_i ||^2_2

    where:
        - ``A`` is the normal (Hermitian) linear operator.
        - ``b`` are the measured data.
        - Each ``damp_i`` is a scalar controlling the strength of ``R_i``.
        - Each ``R_i`` is a regularization linear operator.
        - Each ``bias_i`` is a prior for L2 regularization.

    Notes
    -----
    Setting ``damp`` to ``0.0`` implies solving unregularized problem, regardless
    of provided regularizers and biases. Setting ``damp`` to a non-zero scalar
    without providing regularizers implies standard L2 (Tikhonov) regularization
    (i.g., ``R = I``; ``bias = 0.0``).

    Parameters
    ----------
    A : NDArray[complex | float]
        Matrix representing the linear operator that computes the action of A on a vector.
    b : NDArray[complex | float]
        Right-hand side observation vector.
    damp: float | list[float] | tuple[float] | None, optional
        Regularization strength. If scalar, assume same regularization
        for all the priors. The default is ``0.0``.
    R: NDArray[complex | float] | list[NDArray[complex | float]] | None, optional
        Linear operator for L2 regularization. If not specified, and ``damp != 0.0``,
        this is the identity (Tikhonov regularization).
    bias: NDArray[complex | float] | None, optional
        Bias for L2 regularization (prior image).
        The default is ``None``.
    record_time : bool, optional
        Toggle wheter record runtime (default is ``False``).

    Returns
    -------
    NDArray[complex | float]
        Solution to the problem.

    """
    solver = LSTSQ(
        A,
        b,
        R,
        damp,
        bias,
        record_time,
        verbose,
    )
    return solver.run()


class LSTSQ(App):
    r"""
    Regularized linear least squares for dense problems.

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
    A : NDArray[complex | float]
        Matrix representing the linear operator that computes the action of A on a vector.
    b : NDArray[complex | float]
        Right-hand side observation vector.
    damp: float | list[float] | tuple[float] | None, optional
        Regularization strength. If scalar, assume same regularization
        for all the priors. The default is ``0.0``.
    R: NDArray[complex | float] | list[NDArray[complex | float]] | None, optional
        Linear operator for L2 regularization. If not specified, and ``damp != 0.0``,
        this is the identity (Tikhonov regularization).
    bias: NDArray[complex | float] | None, optional
        Bias for L2 regularization (prior image).
        The default is ``None``.
    record_time : bool, optional
        Toggle wheter record runtime (default is ``False``).

    """

    def __init__(
        self,
        A: NDArray[complex | float],
        b: NDArray[complex | float],
        R: NDArray[complex | float] | list[NDArray[complex | float]] | None = None,
        damp: float | list[float] | tuple[float] | None = None,
        bias: NDArray[complex | float] | None = None,
        verbose: bool = False,
        record_time: bool = False,
    ):
        _alg = _LSTSQ(A, b, R, damp, bias, record_time, verbose)
        super().__init__(_alg, False, False, False)

    def _output(self):
        gc.collect()
        if CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()
        return self.alg.x


# %% utils
class _LSTSQ(Alg):
    def __init__(
        self,
        A: NDArray[complex | float],
        b: NDArray[complex | float],
        R: NDArray[complex | float] | list[NDArray[complex | float]] | None = None,
        damp: float | list[float] | tuple[float] | None = None,
        bias: NDArray[complex | float] | None = None,
        verbose: bool = False,
        record_time: bool = False,
    ):
        A_reg, b_reg = build_extended_system(A, b, R, damp, bias)
        self.A = A_reg
        self.b = b_reg
        self._finished = False
        self._verbose = verbose
        self._record_time = record_time

        super().__init__(max_iter=1)

    def update(self):  # noqa
        if self._record_time:
            timer = Monitor(self.A_reg, self.b_reg)
            timer.start_timer()

        # actual run
        if self._verbose:
            print("LSTSQ start")
        self.x = _lstsq(
            self.A,
            self.b,
        )  # here we let scipy/cupy handle steps.
        if self._verbose:
            print("LSTSQ end")

        # stop timer
        if self._record_time:
            timer.stop_timer()
            self.time = timer.time
            if self._verbose:
                print(f"Elapsed time: {self.time} s")
        self._finished = True

    def _done(self):
        return self._finished


@with_torch
def _lstsq(A, b):
    x = torch.linalg.lstsq(A, b, rcond=None)[0]
    if x.shape[0] == 1:
        x = x[0]
    return x
