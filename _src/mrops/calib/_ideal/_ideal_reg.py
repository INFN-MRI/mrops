"""IDEAL regularization utils."""

__all__ = [
    "median_filter",
    "nonnegative_constraint",
    "WeightedMean",
    "LowPassFilter",
]

from numpy.typing import NDArray
import numpy as np

import scipy.ndimage as ndi

from ..._sigpy import Device, get_array_module
from ...base import fft, ifft


def nonnegative_constraint(
    input: NDArray[float],
) -> tuple[NDArray[float], NDArray[float]]:
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


def median_filter(
    img: NDArray[float], mask: NDArray[bool], size: int = 3
) -> NDArray[float]:
    """
    Apply median filtering for smoothing.

    Parameters
    ----------
    img : NDArray[float]
        Input image.
    mask : NDArray[bool]
        Input mask.
    size : int
        Filter size.

    Returns
    -------
    NDArray[float]
        Smoothed image.

    """
    # Argument checks
    ndim = mask.ndim
    size = np.asarray(ndim * [size])  # Ensure P is an array
    if any(s < 1 or s % 1 != 0 for s in size):
        raise ValueError("Invalid kernel size. Must be positive integers.")
    if not np.isrealobj(img):
        raise ValueError("Median is not well-defined for complex numbers.")
    if len(size) > img.ndim:
        raise ValueError("P has more dimensions than A.")

    # Handle mask
    orig_img = img.copy()
    img = img.astype(np.float32)  # Convert to float to allow NaNs
    img[np.logical_not(mask)] = np.nan  # Exclude masked elements using NaN

    # Apply circular median filter
    img_filt = ndi.median_filter(img, size=size, mode="wrap")

    # Restore original values where mask is False
    img_filt[np.logical_not(mask)] = orig_img[np.logical_not(mask)]

    return img_filt


# %% utils
def _weighted_mean(data, weights):
    """Compute the weighted mean of an array."""
    return (data * weights).sum() / (weights).sum()
