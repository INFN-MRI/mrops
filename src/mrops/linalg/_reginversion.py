"""Sparse and dense linear solvers utils."""

__all__ = ["build_extended_system", "build_extended_square_system"]

from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from .._sigpy import get_device
from .._sigpy.linop import Linop, Identity

from ..interop import aslinearoperator, StackedLinearOperator

from mrinufft._array_compat import get_array_module


def build_extended_system(
    A: Linop,
    b: NDArray,
    Rop: list[Linop],
    lamda: float | list[float],
    bias: NDArray | list[NDArray],
) -> tuple[StackedLinearOperator | NDArray, NDArray]:
    """
    Build the extended system by stacking the original system (A, b)
    with the regularization operators and their respective damping factors and biases.

    Parameters
    ----------
    A : Linop
        The main encoding operator.
    b : NDArray
        The vector of observed data.
    Rop : list[Linop]
        The regularization operators.
    lamda : float | list[float]
        The damping factors for each regularization operator.
    bias : NDArray | list[NDArray]
        The bias terms for each regularization operator.

    Returns
    -------
    A_reg : StackedLinearOperator | NDArray
        The extended linear operator that includes both the encoding operator and regularization operators.
    b_reg : NDArray
        The extended right-hand side vector, including biases for the regularization operators.

    """
    xp = get_array_module(b)
    A, b, Rop, lamda, bias = _preprocess_system(A, b, Rop, lamda, bias)

    if A.__class__.__bases__[0].__name__ == "Linop":
        A = aslinearoperator(A, b)
        Rop = [aslinearoperator(R, b) for R, b in zip(Rop, bias)]

        # Create the stacked linear operator
        A_reg = StackedLinearOperator(A, Rop, lamda)

        # Extend the right-hand side vector with biases
        b_reg = xp.concatenate(
            [b.ravel()] + [l**0.5 * b.ravel() for l, b in zip(lamda, bias)]
        )
    else:
        A_reg = xp.concatenate([A] + [l**0.5 * R for l, R in zip(lamda, Rop)], axis=-2)
        b_reg = xp.concatenate([b] + [l**0.5 * b for l, b in zip(lamda, bias)], axis=-2)

    return A_reg, b_reg


def build_extended_square_system(
    A: Linop,
    b: NDArray,
    Rop: list[Linop],
    lamda: float | list[float],
    bias: NDArray | list[NDArray],
) -> tuple[LinearOperator | NDArray, NDArray]:
    """
    Build the extended system by stacking the original system (A, b)
    with the regularization operators and their respective damping factors and biases.

    Parameters
    ----------
    A : Linop
        The main encoding operator.
    b : NDArray
        The vector of observed data.
    Rop : list[Linop]
        The regularization operators.
    lamda : float | list[float]
        The damping factors for each regularization operator.
    bias : NDArray | list[NDArray]
        The bias terms for each regularization operator.

    Returns
    -------
    AHA_reg : LinearOperator | NDArray
        The extended normal linear operator that includes both the encoding operator and regularization operators.
    AHb_reg : NDArray
        The extended right-hand side vector.

    """
    A, b, Rop, lamda, bias = _preprocess_system(A, b, Rop, lamda, bias)
    AHb = A.H(b)  # initialize right-hand vector

    # Create the stacked normal linear operator
    AHA_reg = A.N
    for l, R in zip(lamda, Rop):
        AHA_reg = AHA_reg + l * R.N

    if A.__class__.__bases__[0].__name__ == "Linop":
        AHA_reg = aslinearoperator(AHA_reg, b)

    # Extend the right-hand side vector with biases
    AHb_reg = AHb.ravel()
    for l, b in zip(lamda, bias):
        AHb_reg += l * b

    return AHA_reg, AHb_reg


def _preprocess_system(A, b, Rop, lamda, bias):
    device = get_device(b)
    xp = device.xp

    # Check if problem is sparse or dense
    if A.__class__.__bases__[0].__name__ == "Linop":
        dense = False
        m, n = A.oshape, A.ishape
    else:
        dense = True
        if A.ndim == 2:
            A = A[None, ...]
            b = b[None, ...]
        batch_size, m, n = A.shape

    # If no regularizers are provided but lamda is a scalar, use identity operator
    if Rop is None:
        if dense:
            with device:
                Rop = [xp.eye(n, dtype=b.dtype)]
        else:
            Rop = [Identity(n)]  # Identity operator (I)
    elif isinstance(Rop, (list, tuple)) is False:
        Rop = [Rop]
    if dense:
        Rop = [R[None, ...] if R.ndim == 2 else R for R in Rop]
        Rop = [xp.repeat(R, batch_size, axis=0) if R.shape[0] == 1 else R for R in Rop]

    # Process damping factors (lamda)
    if isinstance(lamda, (int, float)):  # Scalar case
        lamda = [lamda] * len(
            Rop
        )  # Broadcast to match number of regularization operators
    elif len(lamda) != len(Rop):
        raise ValueError(
            f"Mismatch between number of regularizers ({len(Rop)}) and number of damping factors ({len(lamda)})"
        )

    # Process bias (defaults to zero if None)
    if bias is not None:
        if isinstance(bias, (list, tuple)) is False:
            bias = [bias]
    if bias is None:
        with device:
            bias = [xp.zeros(R.ishape, dtype=b.dtype) for R in Rop]
    elif len(bias) != len(Rop):
        raise ValueError(
            f"Mismatch between number of regularizers ({len(Rop)}) and number of biases ({len(bias)})"
        )

    # Remove regularizers, damping factors, and biases where damping is zero
    filtered_data = [(R, l, b) for R, l, b in zip(Rop, lamda, bias) if l != 0.0]
    if filtered_data:
        Rop, lamda, bias = zip(*filtered_data)  # Unpack back into attributes
    else:
        Rop, lamda, bias = [], [], []

    # Precompute Rop_r.H(bias_r) for each r
    if A.__class__.__bases__[0].__name__ == "Linop":
        bias = [R.H(b) for R, b in zip(Rop, bias)]
    else:
        bias = [R.conj().T @ b for R, b in zip(Rop, bias)]

    return A, b, Rop, lamda, bias
