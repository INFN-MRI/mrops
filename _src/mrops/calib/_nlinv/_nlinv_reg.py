"""Sobolev regularization for NLINV estimation."""

__all__ = ["SobolevOp", "kspace_filter"]

import numpy as np

from numpy.typing import NDArray

from ..._sigpy import Device
from ..._sigpy.linop import Linop, IFFT, Multiply


def SobolevOp(
    shape: list[int] | tuple[int],
    kw: float,
    ell: int,
    device: int | Device | None = None,
) -> Linop:
    """
    Sobolev regularization operator.

    Parameters
    ----------
    shape : list[int] | tuple[int]
        Grid shape ``(nz, ny, nx)`` or ``(ny, nx)``.
    kw: float, optional
        Sobolev filter width. The default is ``220.0``.
    ell: int, optional
        Sobolev filter order. The default is ``32``.
    device : int | Device | None, optional
        Computational device. The default is ``None`` (CPU).

    Returns
    -------
    Linop
        Sobolev regularization linear operator..

    """
    try:
        shape = tuple(shape.tolist())
    except Exception:
        shape = tuple(shape)
    weights = kspace_filter(shape, kw, ell, device)

    # Build operators
    FH = IFFT(shape, axes=tuple(range(-len(shape), 0)))
    W = Multiply(shape, weights)

    return FH * W


# %% util
def kspace_filter(
    shape: list[int] | tuple[int],
    kw: float,
    ell: int,
    device: int | Device | None = None,
) -> NDArray[float]:
    """
    Low pass filter in k-space.

    Used for Sobolev regularization.

    Parameters
    ----------
    shape : list[int] | tuple[int]
        Grid shape ``(nz, ny, nx)`` or ``(ny, nx)``.
    kw: float, optional
        Sobolev filter width. The default is ``220.0``.
    ell: int, optional
        Sobolev filter order. The default is ``32``.
    device : int | Device | None, optional
        Computational device. The default is ``None`` (CPU).

    Returns
    -------
    NDArray[float]
        Low pass filter in k-space of shape ``(*shape)``.

    """
    xp = device.xp if device is not None else np
    with device:
        kgrid = xp.meshgrid(
            *[xp.arange(-n // 2, n // 2, dtype=xp.float32) for n in shape],
            indexing="ij",
        )
    k_norm = sum(ki**2 for ki in kgrid)
    weights = 1.0 / (1 + kw * k_norm) ** (ell / 2)

    return weights.astype(xp.float32)
