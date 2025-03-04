"""Porting of original MATLAB implementation of NLINV."""

__all__ = ["nlinv_python"]

import numpy as np

from numpy.typing import ArrayLike

from ..base import NonLinop

from .._sigpy import get_device
from .._sigpy import linop
from .._sigpy import estimate_shape

from ..base import FFT, IFFT, NUFFT
from ..gadgets import MulticoilOp
from ..solvers import IrgnmCG

def nlinv_python(y, mask, niter):
    """
    Nonlinear inversion for parallel MRI reconstruction.

    Parameters
    ----------
    Y : ndarray (c, y, x)
        k-space measurements.
    n : int
        Number of nonlinear iterations.

    Returns
    -------
    R : ndarray (n, y, x)
        Reconstructed images.
    """
    alpha0 = 1.0
    nc, ny, nx = y.shape

    # Initialization of x-vector
    xhat0 = np.zeros((nc + 1, ny, nx), dtype=np.complex64)
    xhat0[0] = 1.  # Object part
    
    # Initialize Nlinv operator
    _nlinv = CartesianNlinvOp(get_device(y), nc, mask)

    # Normalize data vector
    yscale = 100. / np.linalg.norm(y)
    y = y * yscale

    # initialize variables
    xhat = xhat0.copy()    
    x = _nlinv.W(xhat)
    
    for n in range(niter):
        # Update regularization
        alpha = alpha0 * (2/3)**n
        
        # Update operators
        _nlinv.update(xhat)
        
        # Calculate RHS
        r = _nlinv.DF_n.H.apply(y - _nlinv.F_n.apply(xhat[0])) + alpha * (xhat0 - xhat)
        
        # Calculate CG square matrix operator
        A =  _nlinv.DF_n.H * _nlinv.DF_n + alpha * linop.Identity(_nlinv.DF_n.H.oshape)

        # Conjugate Gradient (CG) initialization
        z = np.zeros_like(r)
        d = np.zeros_like(r)

        dnew = np.vdot(r, r)
        dnot = dnew
        d[:] = r

        for j in range(10):
            # Regularized normal equations
            q = A.apply(d)

            a = dnew / np.real(np.vdot(d, q))
            z += a * d
            r -= a * q

            dold = dnew
            dnew = np.vdot(r, r)

            d = (dnew / dold) * d + r

            if np.sqrt(dnew) < 1.e-2 * dnot:
                break

        # End CG
        xhat += z
        
    # Post-processing
    x = _nlinv.W(xhat) / yscale
    
    # Split output
    rho = x[0]
    smaps = x[1:]
    
    # Normalize
    rho = rho * (smaps.conj() * smaps).sum(axis=0)**0.5

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
    X[0, d < 0.4 ** 2] = 1.0

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
    P[:, ::2] = 1.  # Every other column
    P[:, (y // 2 - 8):(y // 2 + 8)] = 1.  # Center region

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
                *[
                    xp.arange(-n // 2, n // 2, dtype=xp.float32) / n
                    for n in shape
                ],
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
                    * linop.Slice(
                        (n_coils + 1,) + tuple(shape), n + 1
                    )
                    + linop.Multiply(shape, smaps[n])
                    * linop.Slice(
                        (n_coils + 1,) + tuple(shape), 0
                    )
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







