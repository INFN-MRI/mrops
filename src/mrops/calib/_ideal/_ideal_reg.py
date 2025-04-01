"""IDEAL regularization utils."""

__all__ = ["median_smooth", "phase_unwrap", "unswap"]

from numpy.typing import NDArray
import numpy as np

import scipy.ndimage as ndi
from skimage.restoration import unwrap_phase

from scipy.fftpack import fftn, ifftn


def kspace_smoothing(p: np.ndarray, filter_size: int = 3) -> np.ndarray:
    """
    Apply k-space low-pass filtering equivalent to spatial convolution.

    Parameters
    ----------
    p : ndarray
        Input data array of shape (nx, ny, nz).
    filter_size : int, optional
        Size of the spatial-domain box filter (default is 3).

    Returns
    -------
    p_smooth : ndarray
        Smoothed output array of the same shape as `p`.

    Notes
    -----
    - This function applies a hard cutoff low-pass filter in k-space.
    - Equivalent to a (filter_size x filter_size x filter_size) spatial box filter.
    - Assumes `p` is real-valued and applies only real-valued smoothing.

    """
    # Get shape
    nx, ny, nz = p.shape

    # Fourier Transform to k-space
    P_k = fftn(p)

    # Create low-pass filter in k-space (hard cutoff, centered)
    H = np.zeros((nx, ny, nz))
    center = (nx // 2, ny // 2, nz // 2)
    half_size = filter_size // 2  # Half-size of the filter

    H[
        center[0] - half_size : center[0] + half_size + 1,
        center[1] - half_size : center[1] + half_size + 1,
        center[2] - half_size : center[2] + half_size + 1,
    ] = 1

    # Apply filter in k-space
    P_k_smooth = P_k * H

    # Inverse Fourier Transform to get back to spatial domain
    p_smooth = ifftn(P_k_smooth).real

    return p_smooth


def median_smooth(img: NDArray[float], size: int = 3) -> NDArray[float]:
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
    return np.sum(data * weights) / np.sum(weights)
