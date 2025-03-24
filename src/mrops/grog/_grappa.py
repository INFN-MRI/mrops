"""GRAPPA operator training. Adapted for convenience from PyGRAPPA."""

__all__ = ["train"]

import warnings

from numpy.typing import NDArray

import numpy as np
from scipy.linalg import expm, logm

from mrinufft._array_compat import with_numpy

from ..solvers import tikhonov_lstsq


def train(
    train_data: NDArray[complex],
    lamda: float | None = None,
    coords: NDArray[float] | None = None,
) -> dict:
    """
    Train GRAPPA Operator Gridding (GROG) interpolator.

    Parameters
    ----------
    train_data : NDArray[complex]
        Calibration region data of shape ``(nc, nz, ny, nx)`` or ``(nc, ny, nx)``.
        Usually a small portion from the center of kspace.
    lamda : float | None, optional
        Tikhonov regularization parameter.  Set to 0 for no
        regularization. Defaults to ``0.01`` for standard GROG.
        and ``0.0`` for self-calibrating GROG.
    coords : NDArray[float], optional
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT
        (``None`` for Cartesian).

    Returns
    -------
    dict
        Output grog interpolator with keys ``(x, y)`` (single slice 2D)
        or ``(x, y, z)`` (multi-slice or 3D).

    Notes
    -----
    Produces the unit operator described in [1]_.

    This seems to only work well when coil sensitivities are very
    well separated/distinct.  If coil sensitivities are similar,
    operators perform poorly.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Parallel magnetic resonance
           imaging using the GRAPPA operator formalism." Magnetic
           resonance in medicine 54.6 (2005): 1553-1556.

    """
    if lamda is None:
        lamda = 0.01 if coords is None else 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return _train(train_data, lamda, coords)


# %% subroutines
@with_numpy
def _train(train_data, lamda, coords):
    if coords is None:
        ndim = len(train_data.shape) - 1
    else:
        ndim = coords.shape[-1]

    # get grappa operator
    kern = _calc_grappaop(ndim, train_data, lamda, coords)
    Gx = kern["Gx"]
    Gy = kern["Gy"]
    if ndim == 3:
        Gz = kern["Gz"]
    else:
        Gz = None

    return {"x": Gx, "y": Gy, "z": Gz}


def _calc_grappaop(ndim, train_data, lamda, coords):
    train_data = train_data / np.linalg.norm(train_data)
    if coords is not None:
        gz = None
        gy, gx = _radial_grappa_op(train_data, lamda, coords)
    else:
        if ndim == 2:
            gz = None
            gy, gx = _grappa_op_2d(train_data, lamda)
        elif ndim == 3:
            gz, gy, gx = _grappa_op_3d(train_data, lamda)

    return {"Gx": gx, "Gy": gy, "Gz": gz}


def _radial_grappa_op(calib, lamda, coords):
    """Return a 2D GROG operators from radial data."""
    calib = np.moveaxis(calib, 0, -1)
    nr, ns, nc = calib.shape

    # extract x and y components of trajectory
    xcoord = coords[..., 0].T
    ycoord = coords[..., 1].T

    # we need sources (last source has no target!)
    S = calib[:, :-1, ...]

    # and targets (first target has no associated source!)
    T = calib[:, 1:, ...]

    # train the operator
    Gtheta = tikhonov_lstsq(S, T, lamda)
    lGtheta = np.stack([logm(G) for G in Gtheta])
    lGtheta = np.reshape(lGtheta, (nr, nc**2), "F")

    # we now need Gx, Gy.
    dx = np.mean(np.diff(xcoord, axis=0), axis=0)
    dy = np.mean(np.diff(ycoord, axis=0), axis=0)
    dxy = np.concatenate((dx[:, None], dy[:, None]), axis=1)
    dxy = dxy.astype(lGtheta.dtype)

    # solve
    lG = tikhonov_lstsq(dxy, lGtheta, lamda)

    # extract components
    lGx = np.reshape(lG[0, :], (nc, nc))
    lGy = np.reshape(lG[1, :], (nc, nc))

    # take matrix exponential to get from (lGx, lGy) -> (Gx, Gy)
    return expm(lGy), expm(lGx)


def _grappa_op_2d(calib, lamda):
    """Return a 2D GROG operators."""
    calib = np.moveaxis(calib, 0, -1)
    _cx, _cy, nc = calib.shape[:]

    # we need sources (last source has no target!)
    Sy = np.reshape(calib[:, :-1, :], (-1, nc))
    Sx = np.reshape(calib[:-1, ...], (-1, nc))

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Ty = np.reshape(calib[:, 1:, :], (-1, nc))
    Tx = np.reshape(calib[1:, ...], (-1, nc))

    # train the operators:
    Gy = tikhonov_lstsq(Sy, Ty, lamda)
    Gx = tikhonov_lstsq(Sx, Tx, lamda)

    return Gy, Gx


def _grappa_op_3d(calib, lamda):
    """Return 3D GROG operator."""
    calib = np.moveaxis(calib, 0, -1)
    _, _, _, nc = calib.shape[:]

    # we need sources (last source has no target!)
    Sz = np.reshape(calib[:-1, :, :, :], (-1, nc))
    Sy = np.reshape(calib[:, :-1, :, :], (-1, nc))
    Sx = np.reshape(calib[:, :, :-1, :], (-1, nc))

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Tz = np.reshape(calib[1:, :, :, :], (-1, nc))
    Ty = np.reshape(calib[:, 1:, :, :], (-1, nc))
    Tx = np.reshape(calib[:, :, 1:, :], (-1, nc))

    # train the operators:
    Gz = tikhonov_lstsq(Sz, Tz, lamda)
    Gy = tikhonov_lstsq(Sy, Ty, lamda)
    Gx = tikhonov_lstsq(Sx, Tx, lamda)

    return Gz, Gy, Gx
