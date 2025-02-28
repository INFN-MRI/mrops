"""LSMR."""

__all__ = ["LSMR"]

from numpy.typing import ArrayLike

from mrinufft._array_compat import CUPY_AVAILABLE
from mrinufft._array_compat import get_array_module
from mrinufft._array_compat import with_numpy_cupy

from scipy.sparse.linalg import lsmr as scipy_lsmr

if CUPY_AVAILABLE:
    from ._cupy_lsmr import lsmr as cupy_lsmr
    
from .._sigpy.app import App
from .._sigpy.alg import Alg
from .._sigpy.linop import Linop

from ..interop import aslinearoperator


class LSMR(App):
    r"""
    LSMR method.

    Solves the linear system:

    .. math:: A x = b

    where A is a linear operator.

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
            max_iter: int = 10, 
            tol: float = 0.0,
            show_pbar: bool = False,
            leave_pbar: bool = True,
            record_time: bool = False,
        ):
        _alg = _LSMR(A, b, x, max_iter, tol)
        super().__init__(_alg, show_pbar, leave_pbar, record_time)
        
    def _output(self):
        return self.alg.x
    

class _LSMR(Alg):
    def __init__(
            self, 
            A: Linop, 
            b: ArrayLike, 
            x: ArrayLike, 
            max_iter: int = 10, 
            tol: float = 0.0,
        ):
        self.A = aslinearoperator(A, b)
        self.b = b
        self.x = x
        self.tol = tol
        self._finished = False

        super().__init__(max_iter)

    def update(self): # noqa
        shape = self.b.shape
        self.x = _lsmr(
            self.A, self.b.ravel(), self.x, atol=self.tol, maxiter=self.max_iter,
        ) # here we let scipy/cupy handle steps.
        self.b = self.b.reshape(*shape)
        if self.A.batched:
            self.x = self.x.reshape(self.A.batchsize, *self.A.ishape[1:])
        else:
            self.x = self.x.reshape(*self.A.ishape)
        self._finished = True

    def _done(self):
        return self._finished


@with_numpy_cupy
def _lsmr(A, b, x0=None, atol=0, maxiter=None, M=None):
    if x0 is not None:
        x0 = x0.ravel()
    if get_array_module(b).__name__ == "numpy":
        return scipy_lsmr(A, b, x0=x0, atol=atol, maxiter=maxiter)[0]
    else:
        return cupy_lsmr(A, b, x0=x0, atol=atol, maxiter=maxiter)[0]
