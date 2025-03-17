"""GRAPPA operator training. Adapted for convenience from PyGRAPPA."""

__all__ = ["train"]

from numpy.typing import ArrayLike

import numpy as np
from scipy.linalg import expm, logm


from mrinufft._array_compat import with_numpy

from ..solvers import tikhonov_lstsq


@with_numpy
def train(
    train_data: ArrayLike,
    lamda: float = 0.01,
    nsteps: int = 11,
    coords: ArrayLike | None = None,
) -> dict:
    """
    Train GRAPPA Operator Gridding (GROG) interpolator.

    Parameters
    ----------
    train_data : np.ndarray | torch.Tensor
        Calibration region data of shape ``(nc, nz, ny, nx)`` or ``(nc, ny, nx)``.
        Usually a small portion from the center of kspace.
    lamda : float, optional
        Tikhonov regularization parameter.  Set to 0 for no
        regularization. Defaults to ``0.01``.
    nsteps : int, optional
        K-space interpolation grid discretization. Defaults to ``11``
        steps (i.e., ``dk = -0.5, -0.4, ..., 0.0, ..., 0.4, 0.5``)
    coords : ArrayLike, optional
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT
        (``None`` for Cartesian).

    Returns
    -------
    dict
        Output grog interpolator with keys ``(gx, gy)`` (single slice 2D)
        or ``(gx, gy, gz)`` (multi-slice or 3D).

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
    if coords is None:
        ndim = len(train_data.shape) - 1
    else:
        ndim = coords.shape[-1]

    # get grappa operator
    kern = _calc_grappaop(ndim, train_data, lamda, coords)

    # build interpolator
    deltas = (np.arange(nsteps) - (nsteps - 1) // 2) / (nsteps - 1)
    Gx = _weight_grid(kern["Gx"], deltas).astype(np.complex64)  # (nsteps, nc, nc)
    Gy = _weight_grid(kern["Gy"], deltas).astype(np.complex64)  # (nsteps, nc, nc)
    if ndim == 3:
        Gz = _weight_grid(kern["Gz"], deltas).astype(
            np.complex64
        )  # (nsteps, nc, nc), 3D only
    else:
        Gz = None

    return {"x": Gx, "y": Gy, "z": Gz}


# %% subroutines
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
    Sx = np.reshape(calib[:-1, ...], (-1, nc))
    Sy = np.reshape(calib[:, :-1, :], (-1, nc))

    # and we need targets for an operator along each axis (first
    # target has no associated source!)
    Tx = np.reshape(calib[1:, ...], (-1, nc))
    Ty = np.reshape(calib[:, 1:, :], (-1, nc))

    # train the operators:
    Sxh = Sx.conj().T
    lamda0 = lamda * np.linalg.norm(Sxh) / Sxh.shape[0]
    Gx = np.linalg.solve(Sxh @ Sx + lamda0 * np.eye(Sxh.shape[0]), Sxh @ Tx)

    Syh = Sy.conj().T
    lamda0 = lamda * np.linalg.norm(Syh) / Syh.shape[0]
    Gy = np.linalg.solve(Syh @ Sy + lamda0 * np.eye(Syh.shape[0]), Syh @ Ty)

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
    Szh = Sz.conj().T
    lamda0 = lamda * np.linalg.norm(Szh) / Szh.shape[0]
    Gz = np.linalg.solve(Szh @ Sz + lamda0 * np.eye(Szh.shape[0]), Szh @ Tz)

    Syh = Sy.conj().T
    lamda0 = lamda * np.linalg.norm(Syh) / Syh.shape[0]
    Gy = np.linalg.solve(Syh @ Sy + lamda0 * np.eye(Syh.shape[0]), Syh @ Ty)

    Sxh = Sx.conj().T
    lamda0 = lamda * np.linalg.norm(Sxh) / Sxh.shape[0]
    Gx = np.linalg.solve(Sxh @ Sx + lamda0 * np.eye(Sxh.shape[0]), Sxh @ Tx)

    return Gz, Gy, Gx


def _weight_grid(G, weight):
    V, E = np.linalg.eig(G)

    # raise to power along expanded first dim
    V = V ** weight[:, None]
    V = np.apply_along_axis(np.diag, 1, V)

    # put together and return
    return E @ V @ np.linalg.inv(E)
