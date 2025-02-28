"""Adapt SigPy linear operator to scipy / cupyx.scipy."""

__all__ = ["aslinearoperator", "SigpyLinearOperator"]

import numpy as np
from numpy.typing import ArrayLike

from mrinufft._array_compat import CUPY_AVAILABLE
from mrinufft._array_compat import get_array_module
from mrinufft._array_compat import with_numpy_cupy

import scipy.sparse.linalg as spla

if CUPY_AVAILABLE:
    import cupyx.scipy.sparse.linalg as cupy_spla

from .._sigpy.linop import Linop

LinearOperator = spla.LinearOperator


@with_numpy_cupy
def aslinearoperator(A: Linop, input: ArrayLike) -> LinearOperator:
    """
    Convert SigPy Linop to Scipy/Cupy LinearOperator.

    Parameters
    ----------
    A : Linop
        Input Linop.
    input : ArrayLike
        Input dataset. Used to infer device and dtype.

    Returns
    -------
    LinearOperator
        Scipy/Cupy LinearOperator mimicking input Linop.

    """
    if A is None:
        return A

    dtype = input.dtype
    if hasattr(A, "batched"):
        batched = A.batched
    elif hasattr(A, "linops") and hasattr(A.linops[0], "batched"):
        batched = A.linops[0].batched
    else:
        batched = False
        
    if batched:
        batchsize = input.shape[0]
    else:
        batchsize = 1

    if get_array_module(input).__name__ == "numpy":
        return SigpyLinearOperator(A, dtype, batched, batchsize)
    else:
        return CupyLinearOperator(A, dtype, batched, batchsize)


class BaseSigpyLinearOperator:  # noqa
    def __init__(self, linop, dtype=None, batched=False, batchsize=1):
        self.linop = linop
        self.ishape = linop.ishape  # Input shape expected by Linop
        self.oshape = linop.oshape  # Output shape expected by Linop
        self.dtype = dtype
        self.batched = batched
        self.batchsize = batchsize

        # Scipy LinearOperator expects (M, N) shape
        M = np.prod(self.oshape[self.batched :]).item() * batchsize
        N = np.prod(self.ishape[self.batched :]).item() * batchsize
        super().__init__(dtype=self.dtype, shape=(M, N))

    def _matvec(self, x):
        if self.batched:
            x = x.reshape(
                self.batchsize, *self.ishape[1:]
            )  # Reshape flat input to Linop's expected shape
        else:
            x = x.reshape(self.ishape)  # Reshape flat input to Linop's expected shape
        y = self.linop.apply(x)  # Apply the Linop
        return y.ravel()  # Flatten the output

    def _rmatvec(self, y):
        if self.batched:
            y = y.reshape(self.batchsize, *self.oshape[1:])  # Reshape output for adjoint operation
        else:
            y = y.reshape(self.oshape)  # Reshape output for adjoint operation
        x = self.linop.H.apply(y)  # Apply the adjoint Linop
        return x.ravel()  # Flatten again


class SigpyLinearOperator(BaseSigpyLinearOperator, spla.LinearOperator):  # noqa
    """SciPy-compatible LinearOperator wrapper for SigPy Linops."""

    pass


if CUPY_AVAILABLE:

    class CupyLinearOperator(BaseSigpyLinearOperator, cupy_spla.LinearOperator):  # noqa
        """CuPy-compatible LinearOperator wrapper for SigPy Linops."""

        pass

    __all__.append("CupyLinearOperator")
