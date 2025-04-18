"""Toeplitz Operator."""

__all__ = ["ToeplitzOp"]

from numpy.typing import ArrayLike

from .._sigpy import linop
from .._sigpy.linop import Multiply

from ..base import FFT

from ._toep import calc_toeplitz_kernel


class ToeplitzOp(linop.Linop):
    """
    Single coil Fourier Normal operator.

    Parameters
    ----------
    shape : ArrayLike[int] | None, optional
        Input shape. Use ``-1`` to enable broadcasting
        across a particular axis (e.g., ``(-1, Ny, Nx)``).
    coords : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    weights : ArrayLike | None, optional
        Fourier domain density compensation array for NUFFT (``None`` for Cartesian).
        If not provided, does not perform density compensation.
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coords : bool, optional
        Normalize coordinates between -pi and pi. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    """

    def __init__(
        self,
        shape: ArrayLike,
        coords: ArrayLike,
        weights: ArrayLike | None = None,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        normalize_coords: bool = True,
    ):
        ndim = coords.shape[-1]
        fft_axes = tuple(range(-1, -(ndim + 1), -1))

        # Generate PSF kernel
        psf = calc_toeplitz_kernel(
            coords, shape, weights, oversamp, eps, normalize_coords
        )

        # Compose operator
        R = linop.Resize(psf.shape, shape)
        F = FFT(psf.shape, axes=fft_axes)
        P = Multiply(psf.shape, psf)
        self._linops = R.H * F.H * P * F * R
        super().__init__(self._linops.oshape, self._linops.ishape)

    def _apply(self, input):
        return self._linops._apply(input)

    def _normal_linop(self):
        return self

    def _adjoint_linop(self):
        return self
