"""Tikhonov regularized lstsq solver for batched matrices."""

__all__ = ["tikhonov_lstsq"]


from numpy.typing import NDArray

import numpy as np
import numba as nb

from mrinufft._array_compat import with_numpy_cupy


@with_torch
def tikhonov_lstsq(A: NDArray, b: NDArray, lamda: float = 0.0) -> NDArray:
    """
    Batched Tikhonov-regularized least squares using Numpy/Cupy.

    Solves [ A ; sqrt(lambda) * I ] x = [ b ; 0 ] for each batch.

    Parameters
    ----------
    A : NDArray
        Input matrices of shape ``(batch, m, n)``.
    B : NDArray
        Multiple right-hand sides of shape ``(batch, m, k)''.
    lamda : float, optional
        Regularization parameter. The default is ``0.0``.

    Returns
    -------
    X : NDArray
        Solution matrices of shape ``(batch, n, k)``.

    """
    if A.ndim == 2:
        A = A[None, ...]
        b = b[None, ...]
    batch_size, m, n = A.shape
    k = b.shape[-1]

    I = torch.eye(n, dtype=A.dtype, device=A.device)  # Identity matrix (n, n)
    sqrt_lambda_I = (
        torch.sqrt(torch.tensor(lamda, dtype=A.dtype, device=A.device)) * I
    )  # (n, n)

    # Expand to match batch size
    sqrt_lambda_I = sqrt_lambda_I.expand(batch_size, n, n)

    # Stack A and sqrt(lambda) * I along rows
    A_reg = torch.cat([A, sqrt_lambda_I], dim=1)  # (batch, m+n, n)
    b_reg = torch.cat(
        [b, torch.zeros(batch_size, n, k, dtype=A.dtype, device=A.device)], dim=1
    )  # (batch, m+n, k)

    # Solve using PyTorch lstsq (supports batch processing)
    x = torch.linalg.lstsq(A_reg, b_reg, rcond=None)[0]
    if x.shape[0] == 1:
        x = x[0]

    return x
