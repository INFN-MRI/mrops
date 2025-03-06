"""Autocalibration region extraction subroutines."""

__all__ = ["extract_acr"]

import numpy as np

from numpy.typing import NDArray
from mrinufft._array_compat import with_numpy_cupy

from .. import _sigpy

@with_numpy_cupy
def extract_acr(
    data: NDArray,
    cal_width: int = 24,
    ndim: int = None,
    coords: NDArray = None,
    weights: NDArray = None,
    shape: int = None,
) -> NDArray | tuple[NDArray, NDArray, NDArray | None]:
    """
    Extract calibration region from input dataset.

    Parameters
    ----------
    data : NDArray
        Input k-space dataset of shape ``(..., *shape)`` (Cartesian) or
        ``(..., npts)`` (Non Cartesian).
    cal_width : int, optional
        Calibration region size. The default is ``24``.
    ndim : int, optional
        Number of spatial dimensions. The default is ``None``.
        Required for Cartesian datasets.
    coords : NDArray, optional
        K-space trajectory of shape ``(..., npts, ndim)``, normalized between ``(-0.5, 0.5)``.
        Required for Non Cartesian datasets. The default is ``None``.
    weights : NDArray, optional
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
    cal_data : NDArray
        Calibration dataset of shape ``(..., *[cal_width]*ndim)`` (Cartesian) or
        ``(..., cal_width)`` (Non Cartesian).
    cal_coords : NDArray, optional
        Trajectory for calibration dataset of shape ``(..., cal_width, ndim)``.
    cal_weights : NDArray, optional
        Density compensation for calibration dataset of shape ``(..., cal_width)``.

    """
    if coords is None:
        if ndim is None:
            raise ValueError(
                "Please provide number of spatial dimensions for Cartesian datasets"
            )
        shape = list(data.shape[-ndim:])
        return _sigpy.resize(data, data.shape[:-ndim] + ndim * [cal_width])

    else:
        if shape is None:
            raise ValueError("Please provide matrix size for Non Cartesian datasets")

        # get indexes for calibration samples
        cal_width = int(
            np.ceil(cal_width * 2**0.5)
        )  # make sure we can extract a squared cal region later
        cal_idx = np.amax(np.abs(coords), axis=-1) < cal_width / min(shape) / 2

        _data = data[..., cal_idx]
        _coords = coords[..., cal_idx, :]
        if weights is not None:
            _weights = weights[..., cal_idx]
        else:
            _weights = None

        return _data, _coords, _weights