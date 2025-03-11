"""Stacked Cartesian MRI operator."""

__all__ = ["StackedCartesianMR"]

import numpy as np

from numpy.typing import ArrayLike

from .._sigpy import linop

from ._cart_op import CartesianMR
from ._stack import stack

class StackedCartesianMR(linop.Linop):
    """
    Single coil stacked Cartesian MR operator.

    Parameters
    ----------
    shape : ArrayLike[int]
        Input shape ``(nstacks, ny, nx)`` (2D)
        or ``(nstacks, nz, ny, nx)`` (3D).
    mask : ArrayLike[int] | None, optional
        Sampling mask for undersampled imaging.
        Must be shaped ``(nstacks, ny, nx | 1)`` (2D)
        or ``(nstacks, nz, ny, nx | 1)`` (2D).
    n_stack_axes : int, optional
        Number of axis (starting from left) representing stack dimensions.
    axes : ArrayLike[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all spatial axes).
    center : bool, optional
        Toggle centered transform. The default is ``True``.

    """

    def __init__(
        self,
        shape: ArrayLike,
        mask: ArrayLike | None = None,
        n_stack_axes: int = 0,
        axes: ArrayLike | None = None,
        center: bool = True,
    ):
        if len(shape) != 2 + n_stack_axes and len(shape) != 3 + n_stack_axes:
            raise ValueError(
                "shape must be either (*stack_axes, ny, nx) or (*stack_axes, nz, ny, nx)"
            )

        # Get shapes
        nstacks = int(np.prod(shape[:n_stack_axes]))
        image_shape = shape
        image_shape_flat = [nstacks] + list(shape[n_stack_axes:-1])

        # Flatten stack axes
        R = linop.Reshape(image_shape_flat, image_shape)

        # Reshape mask
        if mask is None:
            mask = nstacks * [None]
        else:
            if np.allclose(mask.shape[:-1], shape[:-1]) is False:
                raise ValueError(
                    "mask shape must be either (*stack_axes, ny, nx | 1) or (*stack_axes, nz, ny, nx | 1)"
                )
            mask = mask.reshape(*image_shape_flat)

        # Build list of operators
        _linops = [
            R.H * CartesianMR(image_shape_flat[n], mask[n], axes, center) * R
            for n in range(nstacks)
        ]

        # Stack list of operators
        _linop = stack(*_linops)

        super().__init__(_linop.oshape, _linop.ishape)
        self._linop = _linop

    def _apply(self, input):
        return self._linop._apply(input)

    def _adjoint_linop(self):
        return self._linop.H

    def _normal_linop(self):
        return self._linop.N