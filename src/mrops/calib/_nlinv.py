"""Porting of original MATLAB implementation of NLINV."""

__all__ = ["nlinv_calib"]

import numpy as np

from numpy.typing import ArrayLike

from ..base import NonLinop

from mrinufft._array_compat import with_numpy_cupy
from mrinufft._array_compat import get_array_module

from .._sigpy import get_device, fft
from .._sigpy import linop
from .._sigpy import estimate_shape

from ..base import FFT, IFFT, NUFFT
from ..gadgets import MulticoilOp
from ..solvers import IrgnmCG


@with_numpy_cupy
def nlinv_calib(
    y: ArrayLike,
    ndim: int | None = None,
    mask: ArrayLike | None = None,
    shape: ArrayLike | None = None,
    coords: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    oversamp: float = 1.25,
    eps: float = 1e-3,
    sobolev_width: int = 200,
    sobolev_deg: int = 32,
    max_iter: int = 20,
    cg_iter: int = 10,
    cg_tol: float = 1e-2,
    alpha0: float = 1.0,
    alpha_min: float = 0.0,
    q: float = 2 / 3,
    show_pbar: bool = False,
    leave_pbar: bool = True,
    record_time: bool = False,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Estimate coil sensitivity maps using NLINV.

    Parameters
    ----------
    y : ArrayLike
        Measured k-space data of shape ``(n_coils, ...)``
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
        k-space trajectory for NUFFT (``None`` for Cartesian).
    weights : ArrayLike, optional
        k-space density compensation factors for NUFFT (``None`` for Cartesian).
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
        Toggle wheter record runtime (default is ``False``).

    Returns
    -------
    smaps : ArrayLike
        Coil sensitivity maps of shape ``(n_coils, *shape)``
    image : ArrayLike
        Reconstructed magnetization of shape ``(*shape)``

    """
    xp = get_array_module(y)
    device = get_device(y)
    n_coils = y.shape[0]

    # Determine type of acquisition
    if coords is None:  # Cartesian
        # Check number of dimensions
        if ndim is None:
            raise ValueError(
                "ndim not provided for Cartesian acquisition (either 2 or 3)"
            )
        elif ndim != 2 and ndim != 3:
            raise ValueError("ndim for Cartesian acquisition must be either 2 or 3")

        # Check number of slices
        if len(y.shape) == 3:
            n_slices = 1
        else:
            n_slices = y.shape[1]

        # Work in 3D kspace
        if ndim == 2 and n_slices != 1:
            y = fft(y, axes=(-3,), norm="ortho")

        # Handle mask
        if mask is None:
            if ndim == 2:
                mask = abs(y).reshape(-1, *y.shape[1:])[
                    0
                ]  # assume same (kx, ky) pattern for all slices
            else:
                mask = abs(y[0, ..., 0])[..., None]  # assume (ky, kz) mask
            mask = mask > 0
        mask = mask.astype(xp.float32)

        # Get shape (squeeze if n_slices = 1)
        if n_slices != 1:
            shape = (n_slices,) + tuple(y.shape[-2:])
        else:
            shape = tuple(y.shape[-2:])

        # Build operator
        _nlinv = CartesianNlinvOp(device, n_coils, mask, sobolev_width, sobolev_deg)

    else:  # Non Cartesian
        if shape is None:
            shape = estimate_shape(coords)
        if get_device(coords).id >= 0:
            coords = coords.get()

        # Get number of dimensions
        ndim = coords.shape[-1]

        # Check number of slices
        if ndim == 2:
            if len(y.shape) == len(coords.shape):
                n_slices = 1
            else:
                n_slices = y.shape[1]

            # Work in 3D kspace
            if n_slices != 1:
                y = fft(y, axes=(-3,), norm="ortho")
                coord_z = np.arange(-n_slices // 2, n_slices // 2, dtype=coords.dtype)
                coord_z = np.broadcast_to(coord_z, coords.shape[:-1])[..., None]
                coords = np.concatenate((coords, coord_z), axis=-1)

                shape = (n_slices,) + tuple(shape[-2:])
            else:
                shape = tuple(shape[-2:])

        # If weights are provided, pre-weight k-space data
        if weights is not None:
            y = weights**0.5 * y

        # Build operator
        _nlinv = NonCartesianNlinvOp(
            device,
            n_coils,
            shape,
            coords,
            weights,
            oversamp,
            eps,
            sobolev_width,
            sobolev_deg,
        )

    # Enforce shape as list
    try:
        shape = shape.tolist()
    except Exception:
        shape = list(shape)

    # Initialize guess
    if device.id >= 0:
        with device:
            xhat0 = xp.zeros((n_coils + 1, *shape), dtype=y.dtype)
    else:
        xhat0 = xp.zeros((n_coils + 1, *shape), dtype=y.dtype)
    xhat0[0] = 1.0

    # Normalize data vector
    yscale = 100.0 / np.linalg.norm(y)
    y = y * yscale

    # Run algorithm
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

    # Post-processing
    x = _nlinv.W(xhat) / yscale

    # Split output
    rho = x[0]
    smaps = x[1:]

    # Normalize
    rho = rho * (smaps.conj() * smaps).sum(axis=0) ** 0.5

    return smaps, rho


def simu(x, y):
    """
    Simulate object and coil sensitivities, then apply undersampling.

    Parameters
    ----------
    x, y : int
        Dimensions of the image.

    Returns
    -------
    R : ndarray (4, y, x)
        Simulated undersampled k-space data.
    """
    X = np.zeros((5, y, x), dtype=np.complex64)  # Storage order reversed from MATLAB

    i, j = np.meshgrid(np.arange(y), np.arange(x), indexing="ij")
    d = ((i / y) - 0.5) ** 2 + ((j / x) - 0.5) ** 2

    # Object
    X[0, d < 0.4**2] = 1.0

    # Coil sensitivities
    d1 = ((i / y) - 1.0) ** 2 + ((j / x) - 0.0) ** 2
    d2 = ((i / y) - 1.0) ** 2 + ((j / x) - 1.0) ** 2
    d3 = ((i / y) - 0.0) ** 2 + ((j / x) - 0.0) ** 2
    d4 = ((i / y) - 0.0) ** 2 + ((j / x) - 1.0) ** 2

    X[1] = np.exp(-d1)
    X[2] = np.exp(-d2)
    X[3] = np.exp(-d3)
    X[4] = np.exp(-d4)

    # Undersampling pattern
    P = np.zeros((y, x))
    P[:, ::2] = 1.0  # Every other column
    P[:, (y // 2 - 8) : (y // 2 + 8)] = 1.0  # Center region

    # Simulate k-space data
    return op(P, X), P


def op(P, X):
    """
    Apply forward model operator.

    Parameters
    ----------
    P : ndarray (..., y, x)
        Sampling pattern.
    X : ndarray (..., c+1, y, x)
        Input image data.

    Returns
    -------
    K : ndarray (..., c, y, x)
        Output k-space data.
    """
    K = np.zeros_like(X[..., 1:, :, :], dtype=np.complex64)
    for i in range(X.shape[-3] - 1):
        K[..., i, :, :] = P * myfft(X[..., 0, :, :] * X[..., i + 1, :, :])
    return K


def myfft(x):
    """Apply FFT with correct shifting."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))


# %% util
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
        n_dims = len(shape)

        # Get weighting
        xp = self.device.xp
        with self.device:
            kgrid = xp.meshgrid(
                *[xp.arange(-n // 2, n // 2, dtype=xp.float32) / n for n in shape],
                indexing="ij",
            )
        k_norm = sum(ki**2 for ki in kgrid)
        weights = 1.0 / (1 + kw * k_norm) ** (ell / 2)

        # Build operators
        FH = IFFT(shape, axes=tuple(range(-n_dims, 0)))
        W = linop.Multiply(shape, weights)
        return FH * W

    def _compute_forward(self, xhat):
        """Create forward model operator."""
        x = self.W(xhat)
        smaps = x[1:]
        return MulticoilOp(self._PF, smaps)

    def _compute_jacobian(self, xhat):
        """Compute derivative of forward operator."""
        try:
            shape = tuple(self.matrix_size.tolist())
        except Exception:
            shape = tuple(self.matrix_size)

        # Split input
        x = self.W(xhat)
        rho = x[0]
        smaps = x[1:]
        n_coils = smaps.shape[0]

        # Compute current derivative operator
        # PF * (M * dC_n + dM * C_n for n in range(self.n_coils+1))
        unsqueeze = linop.Reshape([1] + self._PF.oshape, self._PF.oshape)
        DF_n = []
        for n in range(n_coils):
            DF_n.append(
                unsqueeze
                * self._PF
                * (
                    linop.Multiply(shape, rho)
                    * self._W
                    * linop.Slice((n_coils + 1,) + tuple(shape), n + 1)
                    + linop.Multiply(shape, smaps[n])
                    * linop.Slice((n_coils + 1,) + tuple(shape), 0)
                )
            )
        return linop.Vstack(DF_n, axis=0)


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

        # Generate Sampling operator
        P = linop.Multiply(tuple(mask.shape), mask)

        # Generate multicoil FFT
        F = FFT(tuple(mask.shape), axes=tuple(range(-n_dims, 0)))

        return P * F


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

        # NUFFT
        F = NUFFT(
            (self.n_coils,) + shape,
            coords,
            oversamp,
            eps,
        )

        # Preconditioning
        if weights is not None:
            DCF = linop.Multiply(F.oshape, weights**0.5)
        else:
            DCF = linop.Identity(F.oshape)

        return DCF * F  # so that Fadj(input) = NUFFT(DCF(input))
