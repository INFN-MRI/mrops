"""Built-in MR Linear Operators."""

__all__ = ["CartesianMR", "NonCartesianMR"]

from types import SimpleNamespace

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
        Input shape. Use ``-1`` to enable broadcasting
        across a particular axis (e.g., ``(-1, Ny, Nx)``).
    mask : ArrayLike[int] | None, optional
        Sampling mask for undersampled imaging.
    axes :  ArrayLike[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all axes).
    center : bool, optional
        Toggle center iFFT. The default is ``True``.

    """

    def __init__(
        self,
        shape: ArrayLike,
        mask: ArrayLike | None = None,
        axes: ArrayLike | None = None,
        center: bool = True,
    ):
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
        return self._normal


class NonCartesianMR(linop.Linop):
    """
    Single coil Non Cartesian MR operator.

    Parameters
    ----------
    ishape : ArrayLike[int] | None, optional
        Input shape. Use ``-1`` to enable broadcasting
        across a particular axis (e.g., ``(-1, Ny, Nx)``).
    coord : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ndim determines the number of dimensions to apply the nufft.
        ``coord[..., i]`` should be scaled to have its range between
        ``-n_i // 2``, and ``n_i // 2``.
    weights : ArrayLike | None, optional
        k-space density compensation factors for NUFFT (``None`` for Cartesian).
        If not provided, does not perform density compensation.
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coord : bool, optional
        Normalize coordinates between -pi and pi. If ``False``,
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
        plan: SimpleNamespace | None = None,
        normalize_coord: bool = True,
        toeplitz: bool | None = None,
    ):
        ndim = coord.shape[-1]

        # Generate NUFFT operator
        F = NUFFT(ishape, coord, oversamp, eps, plan, normalize_coord)

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
            if ndim == 2:
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
