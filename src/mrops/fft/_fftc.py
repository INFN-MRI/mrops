"""Centered Fast Fourier Transform."""

__all__ = ["fft", "ifft"]

from math import ceil

from numpy.typing import ArrayLike

import torch

from mrinufft._array_compat import with_torch

from .._sigpy import util

@with_torch
def fft(
    input: ArrayLike,
    oshape: ArrayLike | None = None,
    axes: ArrayLike | None = None,
    center: bool = True,
    norm: str | None = "ortho",
) -> ArrayLike:
    """
    FFT function that supports centering.

    Parameters
    ----------
    input : ArrayLike
        Input array.
    oshape : ArrayLike[int] | None, optional
        Output shape. The default is ``None`` (same as ``input``).
    axes :  ArrayLike[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all axes).
    center : bool, optional
        Toggle center FFT. The default is ``True``.
    norm : str | None, optional
        Keyword to specify the normalization mode.
        The default is ``"ortho"``.

    Returns
    -------
    array : ArrayLike
        FFT result of dimension ``oshape``.

    """
    if not torch.is_complex(input):
        input = input.to(torch.complex64)

    if center:
        output = _fftc(input, oshape=oshape, axes=axes, norm=norm)
    else:
        output = torch.fft.fftn(input, s=oshape, dim=axes, norm=norm)

    if torch.is_complex(input) and input.dtype != output.dtype:
        output = output.to(input.dtype, copy=False)

    return output


@with_torch
def ifft(
    input: ArrayLike,
    oshape: ArrayLike | None = None,
    axes: ArrayLike | None = None,
    center: bool = True,
    norm: str | None = "ortho",
) -> ArrayLike:
    """
    IFFT function that supports centering.

    Parameters
    ----------
    input : ArrayLike
        Input array.
    oshape : ArrayLike[int] | None, optional
        Output shape. The default is ``None`` (same as ``input``).
    axes :  ArrayLike[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all axes).
    center : bool, optional
        Toggle center iFFT. The default is ``True``.
    norm : str | None, optional
        Keyword to specify the normalization mode.
        The default is ``"ortho"``.

    Returns
    -------
    array : ArrayLike
        iFFT result of dimension ``oshape``.

    """
    if not torch.is_complex(input):
        input = input.to(torch.complex64)

    if center:
        output = _ifftc(input, oshape=oshape, axes=axes, norm=norm)
    else:
        output = torch.fft.ifftn(input, s=oshape, dim=axes, norm=norm)

    if torch.is_complex(input) and input.dtype != output.dtype:
        output = output.to(input.dtype, copy=False)

    return output


# %% local subroutines
def _fftc(input, oshape=None, axes=None, norm="ortho"):
    ndim = input.ndim
    axes = util._normalize_axes(axes, ndim)

    if oshape is None:
        oshape = input.shape

    tmp = util.resize(input, oshape)
    tmp = torch.fft.ifftshift(tmp, dim=axes)
    tmp = torch.fft.fftn(tmp, dim=axes, norm=norm)
    output = torch.fft.fftshift(tmp, dim=axes)
    return output


def _ifftc(input, oshape=None, axes=None, norm="ortho"):
    ndim = input.ndim
    axes = util._normalize_axes(axes, ndim)

    if oshape is None:
        oshape = input.shape

    tmp = util.resize(input, oshape)
    tmp = torch.fft.ifftshift(tmp, dim=axes)
    tmp = torch.fft.ifftn(tmp, dim=axes, norm=norm)
    output = torch.fft.fftshift(tmp, dim=axes)
    return output


def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    output = coord.copy()
    for i in range(-ndim, 0):
        scale = ceil(oversamp * shape[i]) / shape[i]
        shift = ceil(oversamp * shape[i]) // 2
        output[..., i] *= scale
        output[..., i] += shift

    return output
