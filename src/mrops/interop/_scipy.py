"""Adapt SigPy linear operator to scipy / cupyx.scipy."""

__all__ = [
    "aslinearoperator",
    "stackedlinearoperator",
    "SigpyLinearOperator",
    "StackedLinearOperator",
]

import numpy as np
from numpy.typing import NDArray

from mrinufft._array_compat import CUPY_AVAILABLE
from mrinufft._array_compat import with_numpy_cupy

import scipy.sparse.linalg as spla

if CUPY_AVAILABLE:
    import cupyx.scipy.sparse.linalg as cupy_spla

from .._sigpy import get_array_module, get_device
from .._sigpy.linop import Linop

LinearOperator = spla.LinearOperator


@with_numpy_cupy
def aslinearoperator(A: Linop, input: NDArray) -> LinearOperator:
    """
    Convert SigPy Linop to Scipy/Cupy LinearOperator.

    Parameters
    ----------
    A : Linop
        Input Linop.
    input : NDArray
        Input dataset. Used to infer device and dtype.

    Returns
    -------
    LinearOperator
        Scipy/Cupy LinearOperator mimicking input Linop.

    """
    if A is None:
        return A
    dtype = input.dtype
    if get_array_module(input).__name__ == "numpy":
        return SigpyLinearOperator(A, dtype)
    else:
        return CupyLinearOperator(A, dtype)


def stackedlinearoperator(
    A: LinearOperator, Rop: list[LinearOperator], lamda: list[float]
) -> LinearOperator:
    """
    Stack the encoding operator A with the regularization operators R.

    Attributes
    ----------
    A : LinearOperator
        The encoding operator (A).
    Rop : list of LinearOperators
        List of regularization operators (R).
    lamda : list of floats
        List of damping factors (λ).

    """
    if A is None:
        return A
    if "CUPY" in A.__class__.__name__.upper():
        return CupyStackedLinearOperator(A, Rop, lamda)
    else:
        return StackedLinearOperator(A, Rop, lamda)


# %% LinearOperators
class BaseSigpyLinearOperator:  # noqa
    def __init__(self, linop, dtype=None):
        self.linop = linop
        self.ishape = linop.ishape  # Input shape expected by Linop
        self.oshape = linop.oshape  # Output shape expected by Linop
        self.dtype = dtype

        # Scipy LinearOperator expects (M, N) shape
        M = np.prod(self.oshape).item()
        N = np.prod(self.ishape).item()
        super().__init__(dtype=self.dtype, shape=(M, N))

    def _matvec(self, x):
        x = x.reshape(self.ishape)  # Reshape flat input to Linop's expected shape
        y = self.linop.apply(x)  # Apply the Linop
        return y.ravel()  # Flatten the output

    def _rmatvec(self, y):
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


# %% Stacked LinearOperators
class BaseStackedLinearOperator:
    def __init__(self, A, Rop, lamda):
        self.A = A  # The main encoding operator
        self.Rop = Rop  # Regularization operators
        self.lamda = lamda  # Damping factors
        self.num_reg = len(Rop)  # Number of regularizers

        # Compute the stacked shapes
        oshape = A.shape[0] + sum([R.shape[0] for R in Rop])  # Output size
        ishape = A.shape[0]  # Input size (same as A)

        super().__init__(dtype=A.dtype, shape=(oshape, ishape))

    def _matvec(self, x):
        device = get_device(x)
        xp = device.xp

        # Apply the encoding operator A
        y_A = self.A.matvec(x)
        y_R = []

        # Apply the regularization operators with their damping factors
        for i in range(self.num_reg):
            R = self.Rop[i]
            lamda = self.lamda[i]

            # Regularization term: sqrt(lamda) * R * x_R[i]
            y_R.append(lamda**0.5 * R.matvec(x))

        # Combine results from the encoding and regularization operators
        return xp.concatenate([y_A, *y_R])

    def _rmatvec(self, y):
        y_A = y[: self.A.shape[0]]  # First part corresponds to A
        y_R = y[self.A.shape[0] :]  # Remaining part corresponds to regularization terms

        # Apply the adjoint of the encoding operator A
        x = self.A.rmatvec(y_A)

        # Apply the adjoint of the regularization operators with their damping factors
        offset = 0
        for i in range(self.num_reg):
            R = self.Rop[i]
            lamda = self.lamda[i]

            # Regularization term: sqrt(lamda) * R.H * y_R[i]
            x += lamda**0.5 * R.rmatvec(y_R[offset : offset + R.shape[0]])

            offset += R.shape[0]

        # Combine results from the encoding and regularization operators
        return x


class StackedLinearOperator(BaseStackedLinearOperator, spla.LinearOperator):  # noqa
    """SciPy-compatible LinearOperator wrapper for Stacked operators."""

    pass


if CUPY_AVAILABLE:

    class CupyStackedLinearOperator(
        BaseStackedLinearOperator, cupy_spla.LinearOperator
    ):  # noqa
        """CuPy-compatible LinearOperator wrapper for Stacked Linops."""

        pass

    __all__.append("CupyStackedLinearOperator")
