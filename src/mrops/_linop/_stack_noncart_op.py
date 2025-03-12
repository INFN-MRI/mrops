"""Stacked Non Cartesian MRI operator."""

__all__ = ["StackedNonCartesianMR"]

import numpy as np

from numpy.typing import ArrayLike

from .._sigpy import linop

from ._noncart_op import NonCartesianMR
from ._stack import stack


class StackedNonCartesianMR(linop.Linop):
    """
    Single coil stacked Non Cartesian MR operator.

    Parameters
    ----------
    ishape : ArrayLike[int] | None, optional
        Input shape ``(nstacks, ny, nx)`` (2D) or ``(nstacks, nz, ny, nx)`` (3D).
    coords : ArrayLike
        Fourier domain coordinate array of shape ``(nstacks, ..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    weights : ArrayLike | None, optional
        Fourier domain density compensation array for NUFFT (``None`` for Cartesian).
        If not provided, does not perform density compensation. If provided,
        must be shaped ``coords.shape[:-1]``.
    n_stack_axes : int, optional
        Number of axis (starting from left) representing stack dimensions.
    toeplitz : bool | None, optional
        Use Toeplitz PSF to evaluate normal operator.
        The default is ``True`` for 2D imaging and ``False`` for 3D.
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coord : bool, optional
        Normalize coordinates between ``-pi`` and ``pi``. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    """

    def __init__(
        self,
        ishape: ArrayLike,
        coords: ArrayLike,
        weights: ArrayLike | None = None,
        n_stack_axes: int = 1,
        toeplitz: bool | None = None,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        normalize_coords: bool = True,
    ):
        if len(ishape) != 2 + n_stack_axes and len(ishape) != 3 + n_stack_axes:
            raise ValueError(
                "shape must be either (*stack_axes, ny, nx) or (*stack_axes, nz, ny, nx)"
            )

        # Get shapes
        nstacks = int(np.prod(ishape[:n_stack_axes]))
        image_shape = ishape
        image_shape_flat = [nstacks] + list(ishape[n_stack_axes:-1])
        signal_shape = coords.shape[:-1]
        signal_shape_flat = [nstacks] + coords.shape[n_stack_axes:-1]

        # Flatten stack axes
        Ri = linop.Reshape(image_shape_flat, image_shape)
        Rk = linop.reshape(signal_shape, signal_shape_flat)

        # Reshape coordinates and weights
        coords = coords.reshape(nstacks, *signal_shape_flat, -1)
        if weights is None:
            weights = nstacks * [None]
        else:
            weights = weights.reshape(nstacks, *signal_shape_flat)

        # Build list of operators
        _linops = [
            Rk
            * NonCartesianMR(
                image_shape_flat[n],
                coords[n],
                weights[n],
                oversamp,
                eps,
                normalize_coords,
                toeplitz,
            )
            * Ri
            for n in range(nstacks)
        ]

        # Stack list of operators
        _linop = stack(*_linops)

        super().__init__(_linop.oshape, _linop.ishape)
        self._linop = _linop
        self._nstacks = nstacks
        self._R = Ri

        if toeplitz is None:
            if coords.shape[-1] == 2:
                toeplitz = True
            else:
                toeplitz = False
        self._toeplitz = toeplitz

    def _apply(self, input):
        return self._linop._apply(input)

    def _adjoint_linop(self):
        return self._linop.H

    def _normal_linop(self):
        if self._toeplitz is False:
            return self._linop.N

        # Build list of Toeplitz operators
        nstacks = self._nstacks
        R = self._R
        _linops = [R.H * self._linops.linops[n].linops[1].N * R for n in range(nstacks)]

        # Stack list of Toeplitz operators
        _linop = stack(*_linops)
        _linop.repr_str = "StackedToeplitzOp"

        return linop
