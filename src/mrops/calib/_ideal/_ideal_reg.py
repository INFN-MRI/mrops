"""IDEAL regularization utils."""

__all__ = [
    "median_filter",
    "phase_unwrap",
    "unswap",
    "nonnegative_constraint",
    "WeightedMean",
    "LowPassFilter",
]

from numpy.typing import NDArray
import numpy as np

import scipy.ndimage as ndi
from skimage.restoration import unwrap_phase

from ..._sigpy import Device, get_array_module
from ...base import fft, ifft


def nonnegative_constraint(input: NDArray[float]) -> tuple[NDArray[float], NDArray[float]]:
    """
    Simply returns B0 and R2* separately; Here R2* is rectified.

    Parameters
    ----------
    input : NDArray[float]
        Input array

    Returns
    -------
    output : NDArray[float]
        Rectified input.
    doutput : NDArray[float]
        Derivative adjustment factor.

    """
    xp = get_array_module(input)
    output = xp.abs(input)
    doutput = xp.sign(input) + (input == 0)

    return output, doutput


class WeightedMean:
    """
    Apply weighted mean

    Parameters
    ----------
    weights: NDArray[float]

    """

    def __init__(self, weights):
        self._weights = weights

    def __call__(self, input):
        return _weighted_mean(input, self._weights)


class LowPassFilter:
    """
    Apply k-space low-pass filtering equivalent to a spatial box filter.

    Parameters
    ----------
    device : Device
        Computational device
    shape : list[int] | tuple[int]
        Image shape ``(nz, ny, nx)`` or ``(ny, nx)``
    filter_size : int
        Kernel size

    """

    def __init__(
        self, device: int | Device, shape: list[int] | tuple[int], filter_size: int
    ):
        xp = device.xp
        with device:
            filter_kernel = xp.ones(len(shape) * [filter_size], dtype=xp.complex64)
            self._filter_fft = fft(
                filter_kernel, oshape=shape
            )  # Zero-padding filter to match p

    def __call__(self, input: NDArray[complex | float]) -> NDArray[complex | float]:
        xp = get_array_module(input)
        isreal = xp.isrealobj(input)

        # Bring input to frequency domain
        input_fft = fft(input)

        # Perform element-wise multiplication in frequency domain
        output_fft = input_fft * self._filter_fft

        # Compute inverse FFT to obtain the convolved result
        output = ifft(output_fft)  # Take real part to remove numerical artifacts
        if isreal:
            output = output.real

        return output


def median_filter(img: NDArray[float], size: int = 3) -> NDArray[float]:
    """
    Apply median filtering for smoothing.

    Parameters
    ----------
    img : NDArray[float]
        Input image.
    size : int
        Filter size.

    Returns
    -------
    NDArray[float]
        Smoothed image.

    """
    return ndi.median_filter(img, size=size)


# %% utils
def _weighted_mean(data, weights):
    """Compute the weighted mean of an array."""
    return (data * weights).sum() / (weights).sum()
