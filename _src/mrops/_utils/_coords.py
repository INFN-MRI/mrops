"""Coordinates scaling helper."""

__all__ = ["rescale_coords"]

import numpy as np
from numpy.typing import ArrayLike

from mrinufft._array_compat import with_numpy_cupy

from .._sigpy import get_device


@with_numpy_cupy
def rescale_coords(coords: ArrayLike, amp: float | ArrayLike) -> ArrayLike:
    """
    Rescale Fourier domain coordinates to desired amplitude.

    Parameters
    ----------
    coords : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        Can have arbitrary units or scaling.
    amp : float | ArrayLike
        Output scale. This represent the full dynamic range ``2 * kmax``,
        i.e., output coordinates will be scaled between ``(-0.5 * amp, 0.5 * amp)``.
        If array, must have ``ndim`` elements.

    Returns
    -------
    ArrayLike
        Scaled domain coordinate array of shape ``(..., ndim)``.

    """
    device = get_device(coords)
    cmax = abs(coords).reshape(-1, coords.shape[-1]).max(axis=0)
    if np.isscalar(amp):
        amp = coords.shape[-1] * [amp]
    with device:
        return 0.5 * device.xp.asarray(amp, dtype=coords.dtype) * coords / cmax
