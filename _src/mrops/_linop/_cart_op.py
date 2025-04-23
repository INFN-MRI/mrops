"""Cartesian MRI operator."""

__all__ = ["CartesianMR"]

import numpy as np

from numpy.typing import NDArray

from .._sigpy import linop
from .._sigpy.linop import Multiply

from ..base import FFT


class CartesianMR(linop.Linop):
    """
    Single coil Cartesian MR operator.

    Parameters
    ----------
    shape : list[int] | tuple[int]
        Input shape ``(ny, nx)`` (2D)
        or ``(nz, ny, nx)`` (3D).
    mask : NDArray[bool] | None, optional
        Sampling mask for undersampled imaging.
        Must be shaped ``(ny, nx | 1)`` (2D)
        or ``(nz, ny, nx | 1)`` (2D).
    axes : list[int] | tuple[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all spatial axes).
    center : bool, optional
        Toggle centered transform. The default is ``True``.

    """

    def __init__(
        self,
        shape: list[int] | tuple[int],
        mask: NDArray[bool] | None = None,
        axes: list[int] | tuple[int] | None = None,
        center: bool = True,
    ):
        if len(shape) != 2 and len(shape) != 3:
            raise ValueError("shape must be either (ny, nx) or (nz, ny, nx)")
        if mask is not None and np.allclose(mask.shape[:-1], shape[:-1]) is False:
            raise ValueError(
                "mask shape must be either (ny, nx | 1) or (nz, ny, nx | 1)"
            )

        # Default axes
        if axes is None:
            axes = tuple(range(-len(shape), 0))

        # Generate FFT operator
        F = FFT(shape, axes, center)

        # Undersampled FFT
        if mask is not None:
            PF = Multiply(shape, mask) * F
        else:
            PF = F

        super().__init__(PF.oshape, PF.ishape)
        self._linop = PF

    def _apply(self, input):
        return self._linop._apply(input)

    def _adjoint_linop(self):
        return self._linop.H

    def _normal_linop(self):
        return self._linop.N
