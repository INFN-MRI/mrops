"""Porting of original MATLAB implementation of NLINV."""

__all__ = ["nlinv_calib"]

import math
import numpy as np

from numpy.typing import ArrayLike

from mrinufft._array_compat import with_numpy_cupy
from mrinufft._array_compat import get_array_module

from ..base import NonLinop

from .._sigpy import get_device, resize
from .._sigpy import linop
from .._sigpy import estimate_shape
from .._linop import CartesianMR, NonCartesianMR

from ..base._fftc import fft, ifft
from ..base import IFFT
from ..gadgets import MulticoilOp
from ..solvers import IrgnmCG

from ._acr import extract_acr


@with_numpy_cupy
def nlinv_calib(
    y: ArrayLike,
    cal_width: int | None = None,
    ndim: int | None = None,
    mask: ArrayLike | None = None,
    shape: ArrayLike | None = None,
    coords: ArrayLike | None = None,
    weights: ArrayLike | None = None,
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
    ret_image: bool = False,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Estimate coil sensitivity maps using NLINV.

    Parameters
    ----------
    y : ArrayLike
        Measured k-space data of shape ``(n_coils, ...)``
    cal_width : int
        Size of k-space calibration shape, assuming isotropic matrix.
    ndim : int, optional
        Acquisition dimensionality (2D or 3D). Used for Cartesian only.
    mask : ArrayLike, optional
        Cartesian sampling pattern of the same shape as k-space matrix.
        Used for Cartesian only. If Cartesian and not provided, estimate
        from data (non-zero samples).
    shape : ArrayLike[int], optional
        Image dimensions (e.g., ``(nz, ny, nx)`` for 3D or ``(ny, nx)`` for 2D).
        Used for Non Cartesian only.
    coords : ArrayLike, optional
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT
        (``None`` for Cartesian).
    weights : ArrayLike, optional
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
    ret_image : bool, optional
        Toggle whether returning reconstructed image (default is ``False``).

    Returns
    -------
    smaps : ArrayLike
        Coil sensitivity maps of shape ``(n_coils, *shape)``
    acr : ArrayLike
        Autocalibration k-space region of shape ``(n_coils, *cal_shape)``
    image : ArrayLike, optional
        Reconstructed magnetization of shape ``(*shape)``

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
        NONCART, _nlinv, xhat, yscale, cshape0, oshape, ret_image
    )


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
    NONCART, _nlinv, xhat, yscale, cal_width, oshape, ret_image
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

    # Get GRAPPA training data
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

    if ret_image:
        return smaps, grappa_train, rho
    return smaps, grappa_train


# %% util
def kspace_filter(shape, kw, ell, device):
    xp = device.xp
    with device:
        kgrid = xp.meshgrid(
            *[xp.arange(-n // 2, n // 2, dtype=xp.float32) for n in shape],
            indexing="ij",
        )
    k_norm = sum(ki**2 for ki in kgrid)
    weights = 1.0 / (1 + kw * k_norm) ** (ell / 2)
    return weights.astype(xp.float32)


class BaseNlinvOp(NonLinop):
    def __init__(
        self,
        device: str,
        n_coils: int,
        matrix_size: ArrayLike,
        kw: int = 220.0,
        ell: int = 32,
    ):
        self.device = device
        self.n_coils = n_coils
        self.matrix_size = matrix_size

        # Compute the k-space weighting operator W
        self._W = self._get_weighting_op(kw, ell)

        super().__init__()

    def W(self, input):
        output = input.copy()
        for s in range(1, input.shape[0]):
            output[s] = self._W.apply(input[s])

        return output

    def _get_weighting_op(self, kw, ell):
        try:
            shape = tuple(self.matrix_size.tolist())
        except Exception:
            shape = tuple(self.matrix_size)
        weights = kspace_filter(self.matrix_size, kw, ell, self.device)

        # Build operators
        FH = IFFT(shape, axes=tuple(range(-len(shape), 0)))
        W = linop.Multiply(shape, weights)
        return FH * W

    def _compute_forward(self, xhat):
        """Create forward model operator."""
        x = self.W(xhat)
        smaps = x[1:]
        return MulticoilOp(self._PF, smaps)

    def _compute_jacobian(self, xhat):
        """Compute derivative of forward operator."""
        return _NlinvJacobian(self.matrix_size, self._W, self._PF, self.W(xhat))


class CartesianNlinvOp(BaseNlinvOp):
    def __init__(
        self,
        device: str,
        n_coils: int,
        mask: ArrayLike,
        kw: int = 220.0,
        ell: int = 32,
    ):
        matrix_size = mask.shape
        super().__init__(device, n_coils, matrix_size, kw, ell)
        self._PF = self._get_fourier_op(mask)

    def _get_fourier_op(self, mask):
        try:
            shape = tuple(self.matrix_size.tolist())
        except Exception:
            shape = tuple(self.matrix_size)
        n_dims = len(shape)
        ft_axes = tuple(range(-n_dims, 0))

        return CartesianMR(shape, mask, ft_axes)


class NonCartesianNlinvOp(BaseNlinvOp):
    def __init__(
        self,
        device: str,
        n_coils: int,
        matrix_size: ArrayLike,
        coords: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        kw: int = 220.0,
        ell: int = 32,
    ):
        super().__init__(device, n_coils, matrix_size, kw, ell)
        self._PF = self._get_fourier_op(coords, weights, oversamp, eps)

    def _get_fourier_op(self, coords, weights, oversamp, eps):
        try:
            shape = tuple(self.matrix_size.tolist())
        except Exception:
            shape = tuple(self.matrix_size)

        return NonCartesianMR(shape, coords, weights, True, oversamp, eps)


class _NlinvJacobian(linop.Linop):
    def __init__(self, matrix_size, W, PF, x):
        """Compute derivative of forward operator."""
        try:
            shape = tuple(matrix_size.tolist())
        except Exception:
            shape = tuple(matrix_size)

        # Split input
        rho = x[0]
        smaps = x[1:]
        n_coils = smaps.shape[0]

        # Compute current derivative operator
        # PF * (M * dC_n + dM * C_n for n in range(self.n_coils+1))
        unsqueeze = linop.Reshape([1] + PF.oshape, PF.oshape)
        DF_n = []
        for n in range(n_coils):
            DF_n.append(
                unsqueeze
                * PF
                * (
                    linop.Multiply(shape, rho)
                    * W
                    * linop.Slice((n_coils + 1,) + tuple(shape), n + 1)
                    + linop.Multiply(shape, smaps[n])
                    * linop.Slice((n_coils + 1,) + tuple(shape), 0)
                )
            )

        _linop = linop.Vstack(DF_n, axis=0)

        super().__init__(_linop.oshape, _linop.ishape)
        self._linop = _linop
        self._normal = _NlinvNormal(_linop.ishape, W, PF, x)

    def _apply(self, input):
        return self._linop._apply(input)

    def _normal_linop(self):
        return self._normal

    def _adjoint_linop(self):
        return self._linop.H


class _NlinvNormal(linop.Linop):
    def __init__(self, shape, W, PF, x):
        rho = x[0]
        smaps = x[1:]

        # build
        self._W = W
        self.FHF = PF.N
        self.rho = rho
        self.smaps = smaps
        super().__init__(shape, shape)

    def _apply(self, dxhat):
        xp = get_array_module(dxhat)
        dx = self.W(dxhat)

        # Split
        drho_in = dx[0]
        dsmaps_in = dx[1:]

        # Pre-process Fourier Normal operator input
        _tmp = dsmaps_in * self.rho + self.smaps * drho_in

        # Apply Fourier Normal operator
        _tmp = xp.stack([self.FHF.apply(_el) for _el in _tmp])

        # Post-process Fourier Normal operator output
        drho_out = (self.smaps.conj() * _tmp).sum(axis=0)[None, ...]
        dsmaps_out = self.rho.conj() * _tmp

        return self.Wadjoint(xp.concatenate((drho_out, dsmaps_out), axis=0))

    def W(self, input):
        output = input.copy()
        for s in range(1, input.shape[0]):
            output[s] = self._W.apply(input[s])

        return output

    def Wadjoint(self, input):
        output = input.copy()
        for s in range(1, input.shape[0]):
            output[s] = self._W.H.apply(input[s])

        return output
