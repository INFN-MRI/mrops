"""GROG MRI operator."""

__all__ = ["GrogMR"]

import numpy as np

from numpy.typing import ArrayLike

from .._sigpy import linop
from .._sigpy.linop import Multiply

from .. import grog
from ..base import FFT, MultiIndex
from ..gadgets import BatchedOp


def GrogMR(
    ishape: ArrayLike,
    input: ArrayLike,
    coords: ArrayLike,
    train_data: ArrayLike,
    lamda: float = 0.01,
    nsteps: int = 11,
) -> tuple[ArrayLike, linop.Linop]:
    """
    Single coil GROG MR operator.

    Parameters
    ----------
    ishape : ArrayLike[int]
        Input shape ``(ny, nx)`` (2D)
        or ``(nz, ny, nx)`` (3D).
    coords : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    train_data : np.ndarray | torch.Tensor
        Calibration region data of shape ``(nc, nz, ny, nx)`` or ``(nc, ny, nx)``.
        Usually a small portion from the center of kspace.
    lamda : float, optional
        Tikhonov regularization parameter.  Set to 0 for no
        regularization. Defaults to ``0.01``.
    nsteps : int, optional
        K-space interpolation grid discretization. Defaults to ``11``
        steps (i.e., ``dk = -0.5, -0.4, ..., 0.0, ..., 0.4, 0.5``)

    """
    if len(ishape) != 2 and len(ishape) != 3:
        raise ValueError("shape must be either (ny, nx) or (nz, ny, nx)")
    ndim = coords.shape[-1]

    # train GROG interpolator
    interpolator = grog.train(train_data, lamda, nsteps)

    # perform GROG interpolation
    output, indexes, weights = grog.interp(interpolator, input, coords, ishape)

    # build operators
    I = MultiIndex(ishape, indexes)
    F = FFT(ishape, axes=tuple(range(-ndim, 0)))

    # assemble GROG operator
    G = I * F

    # add batch axes
    batch_axes = input.shape[: -len(indexes.shape[:-1])]
    for ax in batch_axes:
        G = BatchedOp(G, ax)

    # improve representation
    G.repr_str = "GROG Linop"

    return weights * output, G
