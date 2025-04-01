"""NLINV linear operator."""

__all__ = ["CartesianNlinvOp", "NonCartesianNlinvOp"]

from numpy.typing import NDArray

from ..._linop import CartesianMR, NonCartesianMR

from ._common import BaseNlinvOp


class CartesianNlinvOp(BaseNlinvOp):
    """
    Cartesian NLINV nonlinear operator.

    Attributes
    ----------
    device: str | int | Device
        Computational device.
    n_coils: int
        Number of coils.
    mask : NDArray[bool] | None, optional
        Sampling mask for undersampled imaging.
        Must be shaped ``(ny, nx | 1)`` (2D)
        or ``(nz, ny, nx | 1)`` (2D).
    kw: float, optional
        Sobolev filter width. The default is ``220.0``.
    ell: int, optional
        Sobolev filter order. The default is ``32``.

    """

    def __init__(
        self,
        device: str,
        n_coils: int,
        mask: NDArray[bool],
        kw: int = 220.0,
        ell: int = 32,
    ):
        matrix_size = mask.shape
        super().__init__(device, n_coils, matrix_size, kw, ell)
        self._PF = self._get_fourier_op(mask)

    def _get_fourier_op(self, mask):
        """Create forward model operator."""
        try:
            shape = tuple(self.matrix_size.tolist())
        except Exception:
            shape = tuple(self.matrix_size)
        n_dims = len(shape)
        ft_axes = tuple(range(-n_dims, 0))

        return CartesianMR(shape, mask, ft_axes)


class NonCartesianNlinvOp(BaseNlinvOp):
    """
    NonCartesian NLINV nonlinear operator.

    Attributes
    ----------
    device: str | int | Device
        Computational device.
    n_coils: int
        Number of coils.
    matrix_size: list[int] | tuple[int]
        Image matrix size ``(nz, ny, nx)`` or ``(ny, nx)``.
    coords : NDArray[float]
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    weights : NDArray[float] | None, optional
        Fourier domain density compensation array for NUFFT (``None`` for Cartesian).
        If not provided, does not perform density compensation. If provided,
        must be shaped ``coords.shape[:-1]``.
    toeplitz : bool | None, optional
        Use Toeplitz PSF to evaluate normal operator.
        The default is ``True`` for 2D imaging and ``False`` for 3D.
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    kw: float, optional
        Sobolev filter width. The default is ``220.0``.
    ell: int, optional
        Sobolev filter order. The default is ``32``.

    """

    def __init__(
        self,
        device: str,
        n_coils: int,
        matrix_size: list[int] | tuple[int],
        coords: NDArray[float],
        weights: NDArray[float] | None = None,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        kw: int = 220.0,
        ell: int = 32,
    ):
        super().__init__(device, n_coils, matrix_size, kw, ell)
        self._PF = self._get_fourier_op(coords, weights, oversamp, eps)

    def _get_fourier_op(self, coords, weights, oversamp, eps):
        """Create forward model operator."""
        try:
            shape = tuple(self.matrix_size.tolist())
        except Exception:
            shape = tuple(self.matrix_size)

        return NonCartesianMR(shape, coords, weights, True, oversamp, eps)
