"""Conjugate Gradient replacement."""

__all__ = ["ConjugateGradient"]

import gc

from numpy.typing import ArrayLike

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

from ..interop import aslinearoperator


class ConjugateGradient(App):
    r"""
    Conjugate gradient method.

    Solves the linear system:

    .. math:: A x = b

    where A is a Hermitian linear operator.

    Parameters
    ----------
    A : Linop
        Linear operator or function that computes the action of A on a vector.
    b : ArrayLike
        Right-hand side observation vector.
    x : ArrayLike
        Initial guess for the solution.
    P : Linop | None, optional
        Preconditioner function (default is ``None``).
    max_iter : int, optional
        Maximum number of iterations (default is ``10``).
    tol : float, optional
        Tolerance for stopping condition (default is ``0.0``).
    show_pbar : bool, optional
        Toggle whether show progress bar (default is ``False``).
    leave_pbar : bool, optional
        Toggle whether to leave progress bar after finished (default is ``True``).
    record_time : bool, optional
        Toggle wheter record runtime (default is ``False``).

    """

    def __init__(
        self,
        A: Linop,
        b: ArrayLike,
        x: ArrayLike | None = None,
        P: Linop | None = None,
        max_iter: int = 10,
        tol: float = 0.0,
        show_pbar: bool = False,
        leave_pbar: bool = True,
        record_time: bool = False,
    ):
        _alg = _ConjugateGradient(A, b, x, P, max_iter, tol)
        super().__init__(_alg, show_pbar, leave_pbar, record_time)

    def _output(self):
        gc.collect()
        if CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()
        return self.alg.x


class _ConjugateGradient(Alg):
    def __init__(
        self,
        A: Linop,
        b: ArrayLike,
        x: ArrayLike,
        P: Linop | None = None,
        max_iter: int = 10,
        tol: float = 0.0,
    ):
        self.A = aslinearoperator(A, b)
        self.b = b
        self.x = x
        self.P = aslinearoperator(P, b)
        self.tol = tol
        self._finished = False

        super().__init__(max_iter)

    def update(self):  # noqa
        if self.x is None:
            self.x = 0 * self.b

        # get shape
        shape = self.b.shape

        # actual run
        self.x = _cg(
            self.A,
            self.b.ravel(),
            self.x.ravel(),
            tol=self.tol,
            maxiter=self.max_iter,
            M=self.P,
        )  # here we let scipy/cupy handle steps.

        # reshape back
        self.b = self.b.reshape(*shape)
        self.x = self.x.reshape(*shape)

        self._finished = True

    def _done(self):
        return self._finished


@with_numpy_cupy
def _cg(A, b, x0, *, tol=0, maxiter=None, M=None):
    if get_array_module(b).__name__ == "numpy":
        return scipy_cg(A, b, x0, atol=tol, rtol=tol, maxiter=maxiter, M=M)[0]
    else:
        return cupy_cg(A, b, x0, atol=tol, rtol=tol, maxiter=maxiter, M=M)[0]
