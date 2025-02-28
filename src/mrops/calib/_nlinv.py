"""NLINV Operator."""

from ..base import NonLinop

import numpy as np

from .._sigpy import linop
from ..base import FFT, NUFFT
from ..gadgets import MulticoilOp


class NLINV(NonLinop):
    """
    Nonlinear operator for calibrationless parallel MRI reconstruction (NLINV).

    This class models the encoding operator A_n = F * S_n, where:
    - F is either an FFT (Cartesian) or NUFFT (Non-Cartesian).
    - S_n is a pointwise multiplication with coil sensitivity maps.

    The Jacobian dA_n is automatically derived.

    Parameters
    ----------
    N : int
        Number of coil channels.
    matrix_size : tuple of int
        Image dimensions (e.g., (nz, ny, nx) for 3D or (ny, nx) for 2D).
    coords : array-like, optional
        k-space trajectory for NUFFT (None for Cartesian).
    """

    def __init__(
        self, n_coils, matrix_size, coords=None, oversamp=1.25, eps=1e-3, ell=16
    ):
        self.n_coils = n_coils
        self.matrix_size = matrix_size

        # Compute the Fourier operator
        self.A = self._get_fourier_op(coords, oversamp, eps)

        # Compute the k-space weighting operator W
        self.w = self._get_weighting_op(ell)
        self.W = self.w * self._get_cartesian_fft_op()

        # Initial guess: M = ones, C = zeros
        x0 = np.zeros((n_coils + 1,) + matrix_size, dtype=np.complex64)
        x0[0] = 1  # Set M to ones

        super().__init__(x0)

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
        F = self.F

        # Get multicoil Encoding operator: A_n = F * S_n
        A_n = MulticoilOp(F, C)

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
                    * linop.Slice((self.n_coils,) + self.matrix_size, n + 1)
                    + linop.Slice((self.n_coils,) + self.matrix_size, 0)
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

        return linop.Multiply((self.n_coils,) + self.matrix_size, w)

    def _get_cartesian_fft_op(self):
        # Determine number of spatial dimensions (ignoring coil dimension)
        spatial_dims = len(self.matrix_size)
        fft_axes = tuple(range(-spatial_dims, 0))  # Last dimensions are spatial

        return linop.FFT((self.n_coils,) + self.matrix_size, axes=fft_axes)

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
            return linop.NUFFT(
                (self.n_coils,) + self.matrix_size, coords, oversamp, eps
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
