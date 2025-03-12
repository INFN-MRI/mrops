"""Autocalibration region extraction subroutines."""

__all__ = ["extract_acr"]

import numpy as np

from numpy.typing import ArrayLike
from mrinufft._array_compat import with_numpy_cupy

from .. import _sigpy
from .._utils import rescale_coords


@with_numpy_cupy
def extract_acr(
    data: ArrayLike,
    cal_width: int = 24,
    ndim: int = None,
    mask: ArrayLike = None,
    coords: ArrayLike = None,
    weights: ArrayLike = None,
    shape: int = None,
) -> tuple[ArrayLike, ArrayLike | None] | tuple[ArrayLike, ArrayLike, ArrayLike | None]:
    """
    Extract calibration region from input dataset.

    Parameters
    ----------
    data : ArrayLike
        Input k-space dataset of shape ``(..., *shape)`` (Cartesian) or
        ``(..., npts)`` (Non Cartesian).
    cal_width : int, optional
        Calibration region size. The default is ``24``.
    ndim : int, optional
        Number of spatial dimensions. The default is ``None``.
        Required for Cartesian datasets.
    mask : ArrayLike, optional
        Sampling mask for Cartesian datasets of shape ``(..., *shape)``.
    coords : ArrayLike, optional
        Fourier domain coordinate array of shape ``(..., npts, ndim)``.
        Required for Non Cartesian datasets. The default is ``None``.
    weights : ArrayLike, optional
        K-space density compensation of shape ``(..., npts)``. The default is ``None``.
    shape : int, optional
        Matrix size of shape ``(ndim,)``.
        Required for Non Cartesian datasets. The default is ``None``.

    Raises
    ------
    ValueError
        If ``ndim`` is not provided for Cartesian datasets (``trajectory = None``) or
        ``shape`` is not provided for Non Cartesian datasets (``trajectory != None``).

    Returns
    -------
    cal_data : ArrayLike
        Calibration dataset of shape ``(..., *[cal_width]*ndim)`` (Cartesian) or
        ``(..., cal_width)`` (Non Cartesian).
    cal_mask : ArrayLike, optional
        Sampling mask for calibration dataset of shape ``(..., cal_width, ndim)`` (Cartesian).
    cal_coords : ArrayLike, optional
        Trajectory for calibration dataset of shape ``(..., cal_width, ndim)`` (Non Cartesian).
    cal_weights : ArrayLike, optional
        Density compensation for calibration dataset of shape ``(..., cal_width)`` (Non Cartesian).

    """
    if coords is None:
        if ndim is None:
            raise ValueError(
                "Please provide number of spatial dimensions for Cartesian datasets"
            )
        shape = list(data.shape[-ndim:])
        _data = _sigpy.resize(data, list(data.shape[:-ndim]) + ndim * [cal_width])
        if mask is not None:
            _mask = _sigpy.resize(mask, ndim * [cal_width])
            return _data, _mask
        return _data
    else:
        if shape is None:
            raise ValueError("Please provide matrix size for Non Cartesian datasets")

        # get indexes for calibration samples
        coords = rescale_coords(
            coords, shape
        )  # enforce scaling between (-0.5 * npix, 0.5 * npix)
        cal_idx = (coords**2).sum(axis=-1) ** 0.5 <= (0.5 * cal_width)
        cal_idx = cal_idx.reshape(-1, cal_idx.shape[-1])
        cal_idx = np.prod(cal_idx, axis=0).astype(bool)

        # select data
        _data = data[..., cal_idx]
        _coords = coords[..., cal_idx, :]
        if weights is not None:
            _weights = weights[..., cal_idx]
        else:
            _weights = None

        return _data, _coords, _weights
