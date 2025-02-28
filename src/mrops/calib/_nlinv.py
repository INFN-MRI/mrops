"""NLINV Operator."""

__all__ = ["nlinv_calib"]

from numpy.typing import ArrayLike

from ..base import NonLinop

import numpy as np

from mrinufft._array_compat import with_numpy_cupy
from mrinufft._array_compat import get_array_module

from .._sigpy import linop
from .._sigpy import estimate_shape
from ..base import FFT, NUFFT
from ..gadgets import MulticoilOp
from ..solvers import IrgnmCG

@with_numpy_cupy
def nlinv_calib(
        y: ArrayLike,
        shape: ArrayLike | None = None,
        coords: ArrayLike | None = None,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        ell: int = 16,
        max_iter: int = 10, 
        cg_iter: int = 10,
        cg_tol: float = 0.0,
        alpha0: float = 1.0, 
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
    shape : ArrayLike[int]
        Image dimensions (e.g., ``(nz, ny, nx)`` for 3D or ``(ny, nx)`` for 2D).
        Used only for Non Cartesian Datasets.
    coords : ArrayLike, optional
        k-space trajectory for NUFFT (``None`` for Cartesian).
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
        Used only for Non Cartesian Datasets.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
        Used only for Non Cartesian Datasets.
    ell : int, optional
        Sobolev norm order for regularization. The default is ``16``.
        
    Returns
    -------
    smaps : ArrayLike
        Coil sensitivity maps of shape ``(n_coils, *shape)``
    image : ArrayLike
        Reconstructed magnetization of shape ``(*shape)``
        
    """
    n_coils = y.shape[0]
    
    # determine type of acquisition
    if coords is None: # Cartesian
        shape = y.shape[1:]
        NLINVOp = NLINV(n_coils, shape, ell=ell)
    else: # Non Cartesian
        if shape is None:
            shape = estimate_shape(coords)
        NLINVOp = NLINV(n_coils, shape, coords, oversamp, eps, ell)
            
    # Initialize guess
    xp = get_array_module(y)
    if xp.__name__ == "cupy":
        with y.device:
            x0 = xp.zeros((n_coils+1, *shape), dtype=y.dtype)
    else:
        x0 = xp.zeros((n_coils+1, *shape), dtype=y.dtype)
    x0[0] = 1.0
    
    # Run algorithm
    _res = IrgnmCG(
        NLINVOp, 
        y, 
        x0, 
        max_iter, 
        cg_iter,
        cg_tol,
        alpha0, 
        q,
        show_pbar,
        leave_pbar,
        record_time,
        ).run()
    
    # post processing
    smaps = _res[1:]
    rho = _res[0]
    rho = rho * (smaps.conj() * smaps).sum(axis=0)**0.5
    
    return smaps, rho

class NLINV(NonLinop):
    """
    Nonlinear operator for calibrationless parallel MRI reconstruction (NLINV).

    This class models the encoding operator A_n = F * S_n, where:
    - F is either an FFT (Cartesian) or NUFFT (Non-Cartesian).
    - S_n is a pointwise multiplication with coil sensitivity maps.

    The Jacobian dA_n is automatically derived.

    Parameters
    ----------
    n_coils : int
        Number of coil channels.
    matrix_size : ArrayLike[int]
        Image dimensions (e.g., ``(nz, ny, nx)`` for 3D or ``(ny, nx)`` for 2D).
    coords : ArrayLike, optional
        k-space trajectory for NUFFT (``None`` for Cartesian).
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    ell : int, optional
        Sobolev norm order for regularization. The default is ``16``.    
    
    """

    def __init__(
        self, 
        n_coils: int, 
        matrix_size: ArrayLike,
        coords: ArrayLike | None = None,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        ell: int = 16,
    ):
        self.n_coils = n_coils
        self.matrix_size = matrix_size

        # Compute the Fourier operator
        self.A = self._get_fourier_op(coords, oversamp, eps)

        # Compute the k-space weighting operator W
        self.w = self._get_weighting_op(ell)
        self.W = self.w * self._get_cartesian_fft_op()

        super().__init__()

    def _compute_forward(self, x):
        """
        Compute the forward operator G_n(x) for MRI encoding.

        Returns
        -------
        linop.Linop
            Forward model G_n(x) as a matrix-free linear operator.

        """
        C = x[1:]  # Coil sensitivity maps

        # Get single coil Encoding operator (FFT or NUFFT)
        A = self.A

        # Get multicoil Encoding operator: A_n = F * S_n
        A_n = MulticoilOp(A, C)

        # Regularized encoding operator G_n = A_n * W
        G_n = A_n * self.W

        return G_n

    def _compute_jacobian(self, x):
        """
        Compute the Jacobian operator dF(x).

        Returns
        -------
        sp.linop.Linop
            SigPy linear operator representing the Jacobian.
        """
        M = x[0]
        C = x[1:]

        # Get single coil encoding operator (FFT or NUFFT)
        F = self.F

        # PF * (M * dC_n + dM * C_n for n in range(self.n_coils+1))
        DA_n = []
        for n in range(self.n_coils):
            DA_n.append(
                F
                * (
                    linop.Multiply(self.matrix_size, M)
                    * linop.Slice((self.n_coils,) + tuple(self.matrix_size.tolist()), n + 1)
                    + linop.Slice((self.n_coils,) + tuple(self.matrix_size.tolist()), 0)
                    * linop.Multiply(self.matrix_size, C[n])
                )
            )
        DG_n = linop.Vstack(DA_n) * self.W

        return DG_n

    def _get_weighting_op(self, ell):
        """
        Compute the k-space weighting operator W.

        Parameters
        ----------
        ell : int
            Order of the Sobolev norm.

        Returns
        -------
        Linop
            SigPy linear operator representing the k-space weighting.

        """
        kgrid = np.meshgrid(
            *[np.fft.fftfreq(n) for n in self.matrix_size], indexing="ij"
        )
        k_norm = sum(ki**2 for ki in kgrid) ** 0.5
        w = (1 + k_norm) ** (ell / 2)  # l = 16 in the paper

        return linop.Multiply((self.n_coils,) + tuple(self.matrix_size.tolist()), w)

    def _get_cartesian_fft_op(self):
        # Determine number of spatial dimensions (ignoring coil dimension)
        spatial_dims = len(self.matrix_size)
        fft_axes = tuple(range(-spatial_dims, 0))  # Last dimensions are spatial

        return FFT((self.n_coils,) + tuple(self.matrix_size.tolist()), axes=fft_axes)

    def _get_fourier_op(self, coords, oversamp, eps):
        """
        Return the Fourier transform operator (FFT for Cartesian or NUFFT for non-Cartesian).

        Parameters
        ----------
        coords : ArrayLike
            K-space coordinates. If none, assume Cartesian.
        oversamp : float
            Grid oversampling factor for NUFFT.
        else : float
            Target numerical accuracy for NUFFT.

        Returns
        -------
        Linop
            The appropriate Fourier operator.

        """
        if coords is None:
            return self._get_cartesian_fft_op()
        else:
            return NUFFT(
                (self.n_coils,) + tuple(self.matrix_size.tolist()), coords, oversamp, eps
            )

    def get_weighting_op(self):
        """
        Return the k-space weighting operator W.

        This can be used externally for pre-processing initial guesses
        or post-processing the final solution.

        Returns
        -------
        sp.linop.Linop
            The k-space weighting operator.
        """
        return self.W
