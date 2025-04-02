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


def nonnegative_constraint(
    self, input: NDArray[float]
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


def phase_unwrap(img: NDArray[float]) -> NDArray[float]:
    """
    Perform phase unwrapping on the input image.

    Parameters
    ----------
    img : NDArray[float]
        Input wrapped phase image.

    Returns
    -------
    np.ndarray
        Unwrapped phase image.

    """
    return unwrap_phase(img)


def unswap(
    B0: NDArray[float], te: NDArray[float], psif: float, magnitude: NDArray[float]
) -> NDArray[float]:
    """
    Use phase unwrapping to remove aliasing and fat-water swaps.

    Parameters
    ----------
    B0 : NDArray[float]
        Initial B0 field map (Hz).
    te : NDArray[float]
        Echo times (seconds).
    psif : float
        Fat-water frequency shift (rad/s).
    magnitude : NDArray[float]
        Magnitude image (for weighting).

    Returns
    -------
    NDArray[float]
        Corrected B0 field map.

    """
    B0 = B0.copy()

    # Define swap frequencies
    swap = [1 / np.min(np.diff(te)), -psif / (2 * np.pi)]
    swap.append(abs(swap[1] - swap[0]))  # Both (Hz)

    # Remove close swaps
    if min(swap[2] / np.array(swap[:2])) < 0.2:
        swap.pop(2)

    # Perform unwrapping for each swap
    for s in swap:
        B0 /= s
        B0 = phase_unwrap(B0)
        B0 *= s

    # Remove gross aliasing using weighted mean
    alias_freq = 2 * np.pi / np.min(np.diff(te))  # Aliasing frequency (rad/s)
    center_freq = _weighted_mean(
        B0.ravel(), magnitude.ravel()
    )  # Weighted center B0 (rad/s)
    nwraps = round(center_freq / alias_freq)
    B0 -= nwraps * alias_freq

    return B0


# %% utils
def _weighted_mean(data, weights):
    """Compute the weighted mean of an array."""
    return (data * weights).sum() / (weights).sum()
