"""NLINV coil sensitivity map estimation."""

__all__ = ["nlinv_calib"]

import math
import numpy as np

from numpy.typing import NDArray

from mrinufft._array_compat import with_numpy_cupy
from mrinufft._array_compat import get_array_module

from ..._sigpy import estimate_shape
from ..._sigpy import get_device
from ..._sigpy import resize

from ...base import fft, ifft
from ...optimize import IrgnmCG

from .._acr import extract_acr

from ._nlinv_op import CartesianNlinvOp, NonCartesianNlinvOp


@with_numpy_cupy
def nlinv_calib(
    y: NDArray[complex],
    cal_width: int | None = None,
    ndim: int | None = None,
    mask: NDArray[bool] | None = None,
    shape: list[int] | tuple[int] | None = None,
    coords: NDArray[float] | None = None,
    weights: NDArray[float] | None = None,
    oversamp: float = 1.25,
    eps: float = 1e-3,
    sobolev_width: int = 200,
    sobolev_deg: int = 32,
    max_iter: int = 10,
    cg_iter: int = 10,
    cg_tol: float = 1e-2,
    alpha0: float = 1.0,
    alpha_min: float = 0.0,
    q: float = 2 / 3,
    show_pbar: bool = False,
    leave_pbar: bool = True,
    record_time: bool = False,
    lowmem: bool = False,
    ret_cal: bool = True,
    ret_image: bool = False,
) -> tuple[NDArray[complex], NDArray[complex], NDArray[complex]]:
    """
    Estimate coil sensitivity maps using NLINV.

    Parameters
    ----------
    y : NDArray[complex]
        Measured k-space data of shape ``(n_coils, ...)``
    cal_width : int
        Size of k-space calibration shape, assuming isotropic matrix.
    ndim : int, optional
        Acquisition dimensionality (2D or 3D). Used for Cartesian only.
    mask : NDArray[bool], optional
        Cartesian sampling pattern of the same shape as k-space matrix.
        Used for Cartesian only. If Cartesian and not provided, estimate
        from data (non-zero samples).
    shape : list[int] | tuple[int], optional
        Image dimensions (e.g., ``(nz, ny, nx)`` for 3D or ``(ny, nx)`` for 2D).
        Used for Non Cartesian only.
    coords : NDArray[float], optional
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT
        (``None`` for Cartesian).
    weights : NDArray[float], optional
        Fourier domain density compensation array for NUFFT (``None`` for Cartesian).
        If not provided, does not perform density compensation.
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
        Used for Non Cartesian only.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
        Used for Non Cartesian only.
    sobolev_width : int, optional
        Sobolev kernel width, i.e., matrix size of the k-space
        region containing expected coil frequences. The default is ``32``.
    sobolev_deg : int, optional
        Sobolev norm order for regularization. The default is ``16``.
    max_iter : int, optional
        Number of outer (Gauss-Newton) iterations (default is ``10``).
    cg_iter : int, optional
        Number of inner (Conjugate Gradient) iterations (default is ``10``).
    cg_tol : float, optional
         Tolerance for Conjugate Gradient stopping condition (default is ``0.0``).
    alpha0 : float, optional
        Initial regularization parameter (default is ``1.0``).
    alpha_min : float, optional
        Minimum regularization parameter (default is ``1e-6``).
    q : float, optional
        Decay factor for α per outer iteration (default is ``2/3``).
    show_pbar : bool, optional
        Toggle whether show progress bar (default is ``False``).
    leave_pbar : bool, optional
        Toggle whether to leave progress bar after finished (default is ``True``).
    record_time : bool, optional
        Toggle whether record runtime (default is ``False``).
    lowmem : bool, optional
        Toggle whether returning low resolution k-space coil kernels (default is ``False``).
    ret_cal : bool, optional
        Toggle whether returning synthesized calibration region (default is ``True``).
    ret_image : bool, optional
        Toggle whether returning reconstructed image (default is ``False``).

    Returns
    -------
    smaps : NDArray[complex]
        Coil sensitivity maps of shape ``(n_coils, *shape)``.
    acr : NDArray[complex]
        Autocalibration k-space region of shape ``(n_coils, *cal_shape)``.
    image : NDArray[complex], optional
        Reconstructed magnetization of shape ``(*shape)``. Only returned
        if ``ret_image == True``


    """
    xp = get_array_module(y)
    device = get_device(y)
    n_coils = y.shape[0]

    # Setup problem
    if coords is None:
        NONCART, oshape, cshape, cshape0, mask, y, _nlinv = _setup_cartesian(
            y,
            ndim,
            mask,
            cal_width,
            sobolev_width,
            sobolev_deg,
        )
    else:
        (
            NONCART,
            oshape,
            cshape,
            cshape0,
            coords,
            weights,
            y,
            _nlinv,
        ) = _setup_noncartesian(
            y,
            shape,
            coords,
            weights,
            cal_width,
            oversamp,
            eps,
            sobolev_width,
            sobolev_deg,
        )
    cshape0 = list(cshape0) if isinstance(cshape0, (list, tuple)) else cshape0.tolist()
    cshape = list(cshape) if isinstance(cshape, (list, tuple)) else cshape.tolist()
    xhat0 = _initialize_guess(device, n_coils, cshape, xp, y.dtype)

    # Normalize input data
    yscale = 100.0 / np.linalg.norm(y)
    y *= yscale

    # Perform reconstruction
    xhat = IrgnmCG(
        _nlinv,
        y,
        xhat0,
        max_iter,
        cg_iter,
        cg_tol,
        alpha0,
        alpha_min,
        q,
        show_pbar,
        leave_pbar,
        record_time,
    ).run()

    return _postprocess_output(
        NONCART,
        _nlinv,
        xhat,
        yscale,
        cshape0,
        oshape,
        lowmem,
        ret_cal,
        ret_image,
    )


# %% utils
def _setup_cartesian(y, ndim, mask, cal_width, sobolev_width, sobolev_deg):  # noqa
    """Setup for Cartesian acquisition."""
    xp = get_array_module(y)
    if ndim is None or ndim not in {2, 3}:
        raise ValueError("ndim must be 2 or 3 for Cartesian acquisition.")

    # For multi-slice 2D acquisitions, work in 3D
    n_slices = 1 if len(y.shape) == 3 else y.shape[1]
    if ndim == 2 and n_slices != 1:
        y = fft(y, axes=(-3,), norm="ortho")
    if mask is None:
        mask = _estimate_mask(y, ndim, xp)
    ishape = (n_slices,) + tuple(y.shape[-2:]) if n_slices != 1 else tuple(y.shape[-2:])

    # Calibraton width as minimum between matrix size and number of slices
    cal_width = np.min([cal_width, *ishape]).item()

    # Now extract ACR
    y, mask = extract_acr(y, cal_width, ndim, mask)

    # Get calibration shape
    cshape = y.shape[1:]

    # Scale Sobolev width
    n = np.max(ishape)

    # Build Operator
    _nlinv = CartesianNlinvOp(
        get_device(y), y.shape[0], mask, sobolev_width / n**2, sobolev_deg
    )

    return False, ishape, cshape, cshape, mask, y, _nlinv


def _setup_noncartesian(
    y, shape, coords, weights, cal_width, oversamp, eps, sobolev_width, sobolev_deg
):  # noqa
    """Setup for Non-Cartesian acquisition."""
    ndim = coords.shape[-1]
    ishape = estimate_shape(coords) if shape is None else shape

    # For multi-slice 2D acquisitions, work in 3D
    if ndim == 2 and len(y.shape) != len(coords.shape):
        n_slices = y.shape[1]
        y = fft(y, axes=(-3,), norm="ortho")
        coord_z = np.broadcast_to(
            np.arange(-n_slices // 2, n_slices // 2, dtype=coords.dtype),
            coords.shape[:-1],
        )[..., None]
        coords = np.concatenate((coords, coord_z), axis=-1)
        ishape = (n_slices,) + tuple(ishape[-2:])
    elif ndim == 2:
        n_slices = 1
        ishape = tuple(ishape[-2:])

    # Calibraton width as minimum between matrix size and number of slices
    cal_width = np.min([cal_width, *ishape]).item()

    # If we are reconstructing a low res image, extend ACR
    # to make sure corners are properly reconstructed in the target ACR
    osf = 1 if cal_width == min(ishape) else 2**0.5
    cal_width0 = int(np.ceil(cal_width).item())
    cal_width = int(np.ceil(osf * cal_width).item())

    # Now extract ACR
    y, coords, weights = extract_acr(
        y, cal_width=cal_width, coords=coords, weights=weights, shape=ishape
    )

    # Apply DCF
    if weights is not None:
        y *= weights**0.5

    # Get calibration shape
    cshape0 = len(ishape) * [cal_width0]
    cshape = len(ishape) * [cal_width]

    # Scale Sobolev width
    n = np.max(ishape)

    # Build Operator
    _nlinv = NonCartesianNlinvOp(
        get_device(y),
        y.shape[0],
        cshape,
        coords,
        weights,
        oversamp,
        eps,
        sobolev_width / n**2,
        sobolev_deg,
    )

    return True, ishape, cshape, cshape0, coords, weights, y, _nlinv


def _estimate_mask(y, ndim, xp):  # noqa
    """Estimate sampling mask for Cartesian acquisition."""
    if ndim == 2:
        mask = abs(y).reshape(-1, *y.shape[1:])[0]
    else:
        mask = abs(y[0, ..., 0])[..., None]
    return (mask > 0).astype(xp.float32)


def _rescale_coordinates(coords, ishape, shape):  # noqa
    """Rescale k-space trajectory for NUFFT."""
    ndim = coords.shape[-1]
    coords_scale = np.asarray(ishape[-ndim:]) / np.asarray(shape[-ndim:])
    return coords_scale * (2 * math.pi * coords / np.asarray(ishape[-ndim:]))


def _initialize_guess(device, n_coils, shape, xp, dtype):  # noqa
    """Initialize solution guess."""
    xhat0 = xp.zeros((n_coils + 1, *shape), dtype=dtype)
    xhat0[0] = 1.0
    return xhat0


def _postprocess_output(
    NONCART, _nlinv, xhat, yscale, cal_width, oshape, lowmem, ret_cal, ret_image
):  # noqa
    """Post-process results and return sensitivity maps and calibration region."""
    x = _nlinv.W(xhat) / yscale

    # Split image and sensitivities
    rho, smaps = x[0], x[1:]

    # Normalize magnitude
    rss = (smaps.conj() * smaps).sum(axis=0) ** 0.5
    rho = rho * rss
    smaps = smaps / rss

    # Normalize phase
    # phref = smaps[0] / abs(smaps[0])
    # smaps = phref.conj() * smaps

    # Low memory mode: return sensitivity maps k-space kernels:
    if lowmem:
        return fft(smaps, axes=tuple(range(-rho.ndim, 0)), norm="ortho")

    # Get GRAPPA training data
    if ret_cal:
        grappa_train = fft(smaps * rho, axes=tuple(range(-rho.ndim, 0)), norm="ortho")
        if NONCART:
            grappa_shape = list(grappa_train.shape[: -len(cal_width)]) + list(cal_width)
            grappa_train = resize(grappa_train, grappa_shape)

    # Interpolate Coil sensitivities to original matrix size
    smaps_shape = [smaps.shape[0]] + list(oshape[-rho.ndim :])
    smaps = ifft(
        resize(fft(smaps, axes=tuple(range(-rho.ndim, 0)), norm="ortho"), smaps_shape),
        axes=tuple(range(-rho.ndim, 0)),
        norm="ortho",
    )

    if ret_image and ret_cal:
        return smaps, grappa_train, rho
    elif ret_image:
        return smaps, rho
    elif ret_cal:
        return smaps, grappa_train
    return smaps
