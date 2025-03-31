"""Sparse and dense linear solvers utils."""

__all__ = ["build_extended_system", "build_extended_square_system"]

import numpy as np

from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from .._sigpy import get_device
from .._sigpy.linop import Linop, Identity

from ..interop import aslinearoperator, StackedLinearOperator

from mrinufft._array_compat import get_array_module


def build_extended_system(
    A: Linop | NDArray[complex | float],
    b: NDArray[complex | float],
    lamda: float | list[float],
    Rop: (
        Linop | list[Linop] | NDArray[complex | float] | list[NDArray[complex | float]]
    ),
    bias: NDArray[complex | float] | list[NDArray[complex | float]],
) -> tuple[StackedLinearOperator | NDArray[complex | float], NDArray[complex | float]]:
    """
    Build the extended system by stacking the original system (A, b)
    with the regularization operators and their respective damping factors and biases.

    Parameters
    ----------
    A: Linop | NDArray[complex | float],
        The main encoding operator.
    b: NDArray[complex | float],
        The vector of observed data.
    lamda: float | list[float],
        The damping factors for each regularization operator.
    Rop: Linop | list[Linop] | NDArray[complex | float] | list[NDArray[complex | float]],
        The regularization operators.
    bias: NDArray[complex | float] | list[NDArray[complex | float]],
        The bias terms for each regularization operator.

    Returns
    -------
    A_reg : StackedLinearOperator | NDArray[complex | float]
        The extended linear operator that includes both the encoding operator and regularization operators.
    b_reg : NDArray[complex | float]
        The extended right-hand side vector, including biases for the regularization operators.

    """
    xp = get_array_module(b)
    if A.__class__.__bases__[0].__name__ == "Linop":
        A, b, lamda, Rop, bias = _preprocess_sparse_system(A, b, lamda, Rop, bias)

        # Create expanded linearo operator
        A, Rop = aslinearoperator(A, b), [
            aslinearoperator(R, b) for R, b in zip(Rop, bias)
        ]
        A_reg = StackedLinearOperator(A, Rop, lamda)

        # Extend the right-hand side vector with biases
        b_reg = xp.concatenate(
            [b.ravel()] + [l**0.5 * b.ravel() for l, b in zip(lamda, bias)]
        )
    else:
        A, b, lamda, Rop, bias = _preprocess_dense_system(A, b, lamda, Rop, bias)
        A_reg = xp.concatenate([A] + [l**0.5 * R for l, R in zip(lamda, Rop)], axis=-2)
        b_reg = xp.concatenate([b] + [l**0.5 * b for l, b in zip(lamda, bias)], axis=1)

    return A_reg, b_reg


def build_extended_square_system(
    A: Linop | NDArray[complex | float],
    b: NDArray[complex | float],
    lamda: float | list[float],
    Rop: (
        Linop | list[Linop] | NDArray[complex | float] | list[NDArray[complex | float]]
    ),
    bias: NDArray[complex | float] | list[NDArray[complex | float]],
) -> tuple[LinearOperator | NDArray[complex | float], NDArray[complex | float]]:
    """
    Build the extended system by stacking the original system (A, b)
    with the regularization operators and their respective damping factors and biases.

    Parameters
    ----------
    A: Linop | NDArray[complex | float],
        The main encoding operator.
    b: NDArray[complex | float],
        The vector of observed data.
    lamda: float | list[float],
        The damping factors for each regularization operator.
    Rop: Linop | list[Linop] | NDArray[complex | float] | list[NDArray[complex | float]],
        The regularization operators.
    bias: NDArray[complex | float] | list[NDArray[complex | float]],
        The bias terms for each regularization operator.

    Returns
    -------
    AHA_reg : LinearOperator | NDArray[complex | float]
        The extended normal linear operator that includes both the encoding operator and regularization operators.
    AHb_reg : NDArray[complex | float]
        The extended right-hand side vector.

    """
    xp = get_array_module(b)
    if A.__class__.__bases__[0].__name__ == "Linop":
        A, b, lamda, Rop, bias = _preprocess_sparse_system(A, b, lamda, Rop, bias)
        AHb = A.H(b)  # initialize right-hand vector

        # Create the stacked normal linear operator
        AHA_reg = A.N
        for l, R in zip(lamda, Rop):
            AHA_reg = AHA_reg + l * R.N
        AHA_reg = aslinearoperator(AHA_reg, b)

        # Extend the right-hand side vector with biases
        AHb_reg = AHb.ravel()
        for l, b in zip(lamda, bias):
            AHb_reg += l * b
    else:
        A, b, lamda, Rop, bias = _preprocess_dense_system(A, b, lamda, Rop, bias)
        AHb = (
            xp.einsum("bij,bj->bi", A.conj().swapaxes(-1, -2), b)
            if A.shape[0] != 1
            else xp.einsum("ij,bj->bi", A[0].conj().T, b)
        )

        # Create the stacked normal linear operator
        AHA_reg = xp.einsum("bij,bjk->bik", A.conj().swapaxes(-1, -2), A)
        for l, R in zip(lamda, Rop):
            AHA_reg = AHA_reg + l * xp.einsum(
                "bij,bjk->bik", R.conj().swapaxes(-1, -2), R
            )

        # Extend the right-hand side vector with biases
        AHb_reg = AHb
        for l, b in zip(lamda, bias):
            AHb_reg += l * b

    return AHA_reg, AHb_reg


# %% utils
def _preprocess_dense_system(A, b, lamda, Rop, bias):
    device = get_device(b)
    xp = device.xp

    # Get shapes
    if A.ndim == 2:
        A = A[None, ...]
    if b.ndim == 1:
        b = b[None, ...]
    batch_size = b.shape[0]
    m, n = A.shape[1:]
    if A.shape[0] != 1 and A.shape[0] != batch_size:
        raise ValueError(
            "A must be either a batch of matrices with the same batch size as data or a single matrix for the whole dataset"
        )

    # Process regularizers (defaults to [eye] if None)
    with device:
        Rop = [xp.eye(n, dtype=b.dtype)] if Rop is None else Rop
    if isinstance(Rop, (list, tuple)) is False:
        Rop = [Rop]
    Rop = [R[None, ...] if R.ndim == 2 else R for R in Rop]
    Rop = [xp.repeat(R, A.shape[0], axis=0) if R.shape[0] == 1 else R for R in Rop]
    for R in Rop:
        if R.shape[0] != A.shape[0]:
            raise ValueError("Each R in Rop must have the same number of batches as A")

    # Process damping factors (lamda)
    if isinstance(lamda, (int, float)):  # Scalar case
        lamda = [lamda] * len(
            Rop
        )  # Broadcast to match number of regularization operators

    # Process bias (defaults to zero if None)
    with device:
        bias = [xp.zeros(n, dtype=b.dtype) for R in Rop] if bias is None else bias
    if isinstance(bias, (list, tuple)) is False:
        bias = [bias]
    bias = [b[None, ...] if b.ndim == 1 else b for b in bias]
    bias = [xp.repeat(b, batch_size, axis=0) if b.shape[0] == 1 else b for b in bias]

    # Check consistency
    n_damp, n_reg, n_bias = len(lamda), len(Rop), len(bias)
    if np.allclose([n_damp, n_reg, n_bias], n_damp) is False:
        raise ValueError(
            f"Mismatch between number of dampings ({n_damp}), regularizers ({n_reg}) and biases ({n_bias})"
        )

    # Remove regularizers, damping factors, and biases where damping is zero
    filtered_data = [(l, R, b) for l, R, b in zip(lamda, Rop, bias) if l != 0.0]
    if filtered_data:
        lamda, Rop, bias = zip(*filtered_data)  # Unpack back into attributes
    else:
        lamda, Rop, bias = [], [], []

    # Precompute Rop_r.H(bias_r) for each r
    bias = [
        (
            xp.einsum("bij,bj->bi", R.conj().swapaxes(-1, -2), b)
            if R.shape[0] != 1
            else xp.einsum("ij,bj->bi", R[0].conj().T, b)
        )
        for R, b in zip(Rop, bias)
    ]

    if batch_size != 1 and A.shape[0] == 1:
        b = b[..., None].swapaxes(0, -1)  # (1, npts, nbatches)
        bias = [
            b[..., None].swapaxes(0, -1) for b in bias
        ]  # [(1, npts, nbatches)_1, ..., (1, npts, nbatches)_r]

    return A, b, lamda, Rop, bias


def _preprocess_sparse_system(A, b, lamda, Rop, bias):
    device = get_device(b)
    xp = device.xp

    # Get shapes
    _, n = A.oshape, A.ishape

    # Process regularizers (defaults to [Identity] if None)
    if Rop is None:
        Rop = [Identity(n)]  # Identity operator (I)
    if isinstance(Rop, (list, tuple)) is False:
        Rop = [Rop]

    # Process damping factors (lamda)
    if isinstance(lamda, (int, float)):  # Scalar case
        lamda = [lamda] * len(
            Rop
        )  # Broadcast to match number of regularization operators

    # Process bias (defaults to zero if None)
    with device:
        bias = [xp.zeros(n, dtype=b.dtype) for R in Rop] if bias is None else bias
    if isinstance(bias, (list, tuple)) is False:
        bias = [bias]

    # Check consistency
    n_damp, n_reg, n_bias = len(lamda), len(Rop), len(bias)
    if np.allclose([n_damp, n_reg, n_bias], n_damp) is False:
        raise ValueError(
            f"Mismatch between number of dampings ({n_damp}), regularizers ({n_reg}) and biases ({n_bias})"
        )

    # Remove regularizers, damping factors, and biases where damping is zero
    filtered_data = [(l, R, b) for l, R, b in zip(lamda, Rop, bias) if l != 0.0]
    if filtered_data:
        lamda, Rop, bias = zip(*filtered_data)  # Unpack back into attributes
    else:
        lamda, Rop, bias = [], [], []

    # Precompute Rop_r.H(bias_r) for each r
    bias = [R.H(b) for R, b in zip(Rop, bias)]

    return A, b, lamda, Rop, bias
