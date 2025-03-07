"""Toeplitz kernel calculators."""

__all__ = ["calc_toeplitz_kernel"]

from numpy.typing import ArrayLike
from mrinufft._array_compat import with_numpy_cupy

from .. import _sigpy

from ..base._fftc import fft
from ..base._nufft import nufft, nufft_adjoint


@with_numpy_cupy
def calc_toeplitz_kernel(
    coord: ArrayLike,
    shape: ArrayLike,
    weights: ArrayLike = None,
    oversamp: float = 1.25,
    eps: float = 1e-6,
    normalize_coord: bool = True,
):
    """
    Toeplitz PSF for fast Normal non-uniform Fast Fourier Transform.

    While fast, this is more memory intensive.

    Parameters
    ----------
    coord : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ndim determines the number of dimensions to apply the nufft.
        ``coord[..., i]`` should be scaled to have its range between
        ``-n_i // 2``, and ``n_i // 2``.
    shape : ArrayLike[int] | None, optional
        Shape of the form ``(..., n_{ndim - 1}, ..., n_1, n_0)``.
        The default is ``None`` (estimated from ``coord``).
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coord : bool, optional
        Normalize coordinates between -pi and pi. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    Returns
    -------
    ArrayLike
        Signal domain data of shape
        ``input.shape[:-ndim] + coord.shape[:-1]``.

    """
    xp = _sigpy.get_array_module(coord)
    with _sigpy.get_device(coord):
        ndim = coord.shape[-1]
        shape = xp.asarray(shape[-ndim:])

        # Get oversampling (2 for Non Uniform axes, 1 for Cartesian Grid)
        _coord = xp.stack(
            [0.5 * coord[..., n] / xp.max(xp.abs(coord[..., n])) for n in range(ndim)],
            axis=-1,
        )
        _coord = shape * _coord
        _coord = xp.stack(
            [_coord[..., n] - xp.min(_coord[..., n]) for n in range(ndim)], axis=-1
        )
        osf = [
            1 if xp.allclose(xp.round(_coord[..., n]), _coord[..., n]) else 2
            for n in range(ndim)
        ]
        osf = xp.asarray(osf)
        os_shape = (osf * shape).tolist()

        # Create test data
        idx = [slice(None)] * len(os_shape)
        for k in range(-1, -(ndim + 1), -1):
            idx[k] = os_shape[k] // 2
        d = xp.zeros(os_shape, dtype=xp.complex64)
        d[tuple(idx)] = 1

        # Generate DCF
        if weights is None:
            weights = xp.ones_like(coord[..., 0])

        # Get Point Spread Function
        psf = nufft(d, coord, oversamp, eps, normalize_coord)
        psf = nufft_adjoint(
            weights * psf, coord, os_shape, oversamp, eps, normalize_coord
        )

        # Kernel is FFT of PSF
        fft_axes = tuple(range(-1, -(ndim + 1), -1))
        psf = fft(psf, axes=fft_axes, norm=None) * (2**ndim)

        return psf
