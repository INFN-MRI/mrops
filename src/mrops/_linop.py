"""Built-in MR Linear Operators."""

__all__ = [
    "CartesianMR",
    "StackedCartesianMR",
    "NonCartesianMR",
    "StackedNonCartesianMR",
]

import numpy as np

from numpy.typing import ArrayLike

from ._sigpy import linop
from ._sigpy.linop import Multiply

from .base import FFT, NUFFT

from .toep import ToeplitzOp


class CartesianMR(linop.Linop):
    """
    Single coil Cartesian MR operator.

    Parameters
    ----------
    shape : ArrayLike[int]
        Input shape ``(ny, nx)`` (2D)
        or ``(nz, ny, nx)`` (3D).
    mask : ArrayLike[int] | None, optional
        Sampling mask for undersampled imaging.
        Must be shaped ``(ny, nx | 1)`` (2D)
        or ``(nz, ny, nx | 1)`` (2D)
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
        axes: ArrayLike | None = None,
        center: bool = True,
    ):
        if len(shape) != 2 or len(shape) != 3:
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
        or ``(nstacks, nz, ny, nx | 1)`` (2D)
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
        axes: ArrayLike | None = None,
        center: bool = True,
    ):
        if len(shape) != 3 or len(shape) != 4:
            raise ValueError(
                "shape must be either (nstacks, ny, nx) or (nstacks, nz, ny, nx)"
            )

        # Build operators
        nstacks = int(shape[0])
        if mask is None:
            mask = nstacks * [None]
        elif np.allclose(mask.shape[:-1], shape[:-1]) is False:
            raise ValueError(
                "mask shape must be either (nstacks, ny, nx | 1) or (nstacks, nz, ny, nx | 1)"
            )
        _linops = [CartesianMR(shape[n], mask[n], axes, center) for n in range(nstacks)]

        self._linop = _as_stacked(*_linops)
        super().__init__(self._linop.oshape, self._linop.ishape)

    def _apply(self, input):
        return self._linop._apply(input)

    def _adjoint_linop(self):
        return self._linop.H

    def _normal_linop(self):
        return self._linop.N


class NonCartesianMR(linop.Linop):
    """
    Single coil Non Cartesian MR operator.

    Parameters
    ----------
    ishape : ArrayLike[int] | None, optional
        Input shape ``(ny, nx)`` (2D) or ``(nz, ny, nx)`` (3D).
    coord : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    weights : ArrayLike | None, optional
        k-space density compensation factors for NUFFT (``None`` for Cartesian).
        If not provided, does not perform density compensation. If provided,
        must be shaped ``coord.shape[:-1]``
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coord : bool, optional
        Normalize coordinates between ``-pi`` and ``pi``. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.
    toeplitz : bool | None, optional
        Use Toeplitz PSF to evaluate normal operator.
        The default is ``True`` for 2D imaging and ``False`` for 3D.

    """

    def __init__(
        self,
        ishape: ArrayLike,
        coord: ArrayLike,
        weights: ArrayLike | None = None,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        normalize_coord: bool = True,
        toeplitz: bool | None = None,
    ):
        if len(ishape) != 2 or len(ishape) != 3:
            raise ValueError("shape must be either (ny, nx) or (nz, ny, nx)")

        # Generate NUFFT operator
        F = NUFFT(ishape, coord, oversamp, eps, normalize_coord=normalize_coord)

        # Density compensation
        if weights is not None:
            PF = Multiply(F.oshape, weights**0.5) * F
        else:
            PF = F

        super().__init__(PF.oshape, PF.ishape)
        self._linop = PF
        self._shape = ishape
        self._coord = coord
        self._weights = weights
        self._oversamp = oversamp
        self._eps = eps
        self._normalize_coord = normalize_coord

        if toeplitz is None:
            if coord.shape[-1] == 2:
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
            return self._linop.H * self._linop

        return ToeplitzOp(
            self._shape,
            self._coord,
            self._weights,
            self._oversamp,
            self._eps,
            self._normalize_coord,
        )


class StackedNonCartesianMR(linop.Linop):
    """
    Single coil stacked Non Cartesian MR operator.

    Parameters
    ----------
    ishape : ArrayLike[int] | None, optional
        Input shape ``(nstacks, ny, nx)`` (2D) or ``(nstacks, nz, ny, nx)`` (3D).
    coord : ArrayLike
        Fourier domain coordinate array of shape ``(nstacks, ..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    weights : ArrayLike | None, optional
        k-space density compensation factors for NUFFT (``None`` for Cartesian).
        If not provided, does not perform density compensation. If provided,
        must be shaped ``coord.shape[:-1]``
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coord : bool, optional
        Normalize coordinates between ``-pi`` and ``pi``. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.
    toeplitz : bool | None, optional
        Use Toeplitz PSF to evaluate normal operator.
        The default is ``True`` for 2D imaging and ``False`` for 3D.

    """

    def __init__(
        self,
        ishape: ArrayLike,
        coord: ArrayLike,
        weights: ArrayLike | None = None,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        normalize_coord: bool = True,
        toeplitz: bool | None = None,
    ):
        if len(ishape) != 3 or len(ishape) != 4:
            raise ValueError(
                "shape must be either (nstacks, ny, nx) or (nstacks, nz, ny, nx)"
            )

        # Build operators
        nstacks = int(coord.shape[0])
        if weights is None:
            weights = nstacks * [None]
        _linops = [
            NonCartesianMR(
                ishape[n],
                coord[n],
                weights[n],
                oversamp,
                eps,
                normalize_coord,
                toeplitz,
            )
            for n in range(nstacks)
        ]

        self._linop = _as_stacked(*_linops)
        super().__init__(self._linop.oshape, self._linop.ishape)

    def _apply(self, input):
        return self._linop._apply(input)

    def _adjoint_linop(self):
        return self._linop.H

    def _normal_linop(self):
        return self._linop.N


# %% local utils
def _as_stacked(*operators):
    nstacks = len(operators)
    shape = operators[0].shape
    _unsqueeze = linop.Reshape((1,) + tuple(shape), tuple(shape))
    _linops = [
        _unsqueeze * operators[n] * linop.Slice(shape, n) for n in range(nstacks)
    ]
    return linop.Vstack(_linops, axis=0)
