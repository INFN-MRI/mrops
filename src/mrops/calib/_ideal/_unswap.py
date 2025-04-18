"""IDEAL phase unwrapping routines."""

__all__ = ["Unswap"]

import warnings

from numpy.typing import NDArray
import numpy as np

from skimage.restoration import unwrap_phase
from mrinufft._array_compat import with_numpy

from ..._sigpy import get_device, to_device, resize
from ...base import fft, ifft

from ._ideal_reg import median_filter


class Unswap:
    """
    Use phase unwrapping to remove aliasing and fat-water swaps.

    Parameters
    ----------
    te : NDArray[float]
        Echo times (seconds).
    chemshift : float
        Fat-water frequency shift (rad/s).
    mask : NDArray[bool]
        Mask for median filtering
    medfilt_size : int
        Size of median filter.
    frequency_width : int | list[int] | tuple[int]
        Size of low frequency region. If scalar, assumes isotropic.

    """

    def __init__(
        self,
        te: NDArray[float],
        chemshift: float,
        mask: NDArray[bool],
        medfilt_size: int = 3,
        frequency_width: int = 64,
    ):
        ndim = mask.ndim
        if ndim == 3 and mask.shape[0] < frequency_width:
            ndim = 2
        frequency_width = np.min([frequency_width] + list(mask.shape[-ndim:]))

        self._filter = fermi(ndim, mask.shape[-1], frequency_width)
        self._cal_shape = self._filter.shape[:-ndim] + tuple(ndim * [frequency_width])
        self._ndim = ndim
        self._te = te
        self._chemshift = chemshift
        self._medfilt_size = medfilt_size

        # _magnitude = fft(magnitude, axes=list(range(-self._ndim, 0)), norm="ortho")
        # _magnitude = resize(_magnitude, self._cal_shape)
        # self._magnitude = ifft(_magnitude, axes=list(range(-self._ndim, 0)), norm="ortho").real
        self._mask = mask

    def __call__(self, input):
        shape = input.shape

        # 1) get B0 part
        B0 = input.real

        # 2) downsample B0 part
        # _B0 = fft(B0, axes=list(range(-self._ndim, 0)), norm="ortho")
        # _B0 = resize(_B0, self._cal_shape)
        # B0 = ifft(_B0, axes=list(range(-self._ndim, 0)), norm="ortho").real

        # 3) unswap downsampled B0 map + median filter
        B0 = unswap(B0, self._te, self._chemshift, self._mask, self._medfilt_size)

        # 4) apply lowpass filter to suppress ringing when upsampling
        # _B0 = fft(B0, axes=list(range(-self._ndim, 0)), norm="ortho")
        # _B0 = self._filter * resize(_B0, shape)
        # B0 = ifft(_B0, axes=list(range(-self._ndim, 0)), norm="ortho").real

        # 5) replace
        return B0 + 1j * input.imag


# %% utils
def fermi(ndim, size, width=None):
    if width is None:
        width = size

    # get radius
    radius = int(width // 2)

    # build grid, normalized so that u = 1 corresponds to window FWHM
    R = [
        np.arange(int(-size // 2), int(size // 2), dtype=np.float32)
        for n in range(ndim)
    ]

    # get transition width
    T = width / 128

    # build center-out grid
    R = np.meshgrid(*R, indexing="xy")
    R = np.stack(R, axis=0)
    R = (R**2).sum(axis=0) ** 0.5

    # build filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filt = 1 / (1 + np.exp((R - radius)) / T)
        filt /= filt.max()

    return np.nan_to_num(filt, posinf=0.0, neginf=0.0)


def unswap(B0, te, chemshift, mask, medfilt_size):
    device = get_device(B0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        B0 = _unswap(B0, te, chemshift, mask, medfilt_size)

    return to_device(B0, device)


@with_numpy
def _unswap(B0, te, chemshift, mask, medfilt_size):
    B0 = B0.copy()

    # Define swap frequencies
    swap = [1 / np.min(np.diff(te)), -chemshift / (2 * np.pi)]
    swap.append(abs(swap[1] - swap[0]))  # Both (Hz)

    # Remove close swaps
    if min(swap[2] / np.asarray(swap[:2])) < 0.2:
        swap.pop(2)

    # Perform unwrapping for each swap
    for s in swap:
        _s = np.asarray(s, dtype=np.float32)
        B0 /= _s
        B0 = unwrap_phase(np.ma.masked_array(B0, mask=np.logical_not(mask))).data
        B0 *= _s

    # Remove gross aliasing using weighted mean
    alias_freq = 2 * np.pi / np.min(np.diff(te))  # Aliasing frequency (rad/s)
    center_freq = B0[mask].mean()
    nwraps = round(center_freq / alias_freq)
    B0 -= nwraps * alias_freq

    # Perform medianl filtering
    B0 = median_filter(B0, mask, medfilt_size)

    return B0
