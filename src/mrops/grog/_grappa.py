"""GRAPPA operator training. Adapted for convenience from PyGRAPPA."""

__all__ = ["grog_train"]

from numpy.typing import ArrayLike

import numpy as np

from mrinufft._array_compat import with_numpy


@with_numpy
def grog_train(train_data: ArrayLike, lamda: float = 0.01, nsteps: int = 11) -> dict:
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
    # get number of spatial dimes
    ndim = len(train_data.shape) - 1

    # get grappa operator
    kern = _calc_grappaop(ndim, train_data, lamda)

    # build interpolator
    deltas = (np.arange(nsteps) - (nsteps - 1) // 2) / (nsteps - 1)
    Gx = _weight_grid(kern["Gx"], deltas)  # (nsteps, nc, nc)
    Gy = _weight_grid(kern["Gy"], deltas)  # (nsteps, nc, nc)
    if ndim == 3:
        Gz = _weight_grid(kern["Gz"], deltas)  # (nsteps, nc, nc), 3D only
    else:
        Gz = None

    return {"x": Gx, "y": Gy, "z": Gz}


# %% subroutines
def _calc_grappaop(ndim, train_data, lamda):
    if ndim == 2:
        train_data = train_data[:, None, :, :].copy()

    # compute kernels
    if ndim == 2:
        gz = None
        gy, gx = _grappa_op_2d(train_data, lamda)
    elif ndim == 3:
        gz, gy, gx = _grappa_op_3d(train_data, lamda)

    return {"Gx": gx, "Gy": gy, "Gz": gz}


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

    return Gx, Gy


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
    Szh = Sz.conj().permute(1, 0)
    lamda0 = lamda * np.linalg.norm(Szh) / Szh.shape[0]
    Gz = np.linalg.solve(Szh @ Sz + lamda0 * np.eye(Szh.shape[0]), Szh @ Tz)

    Syh = Sy.conj().permute(1, 0)
    lamda0 = lamda * np.linalg.norm(Syh) / Syh.shape[0]
    Gy = np.linalg.solve(Syh @ Sy + lamda0 * np.eye(Syh.shape[0]), Syh @ Ty)

    Sxh = Sx.conj().permute(1, 0)
    lamda0 = lamda * np.linalg.norm(Sxh) / Sxh.shape[0]
    Gx = np.linalg.solve(Sxh @ Sx + lamda0 * np.eye(Sxh.shape[0]), Sxh @ Tx)

    return Gz.clone(), Gy.clone(), Gx.clone()


def _weight_grid(A, weight):
    L, V = np.linalg.eig(A)

    # raise to power along expanded first dim
    L = L[None, ...] ** weight[:, None, None]

    # unsqueeze batch dimension for V
    V = V[None, ...]

    # put together and return
    return V @ np.diagonal(L) @ np.linalg.inv(V)
