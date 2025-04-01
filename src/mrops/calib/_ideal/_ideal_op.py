"""IDEAL Nonlinear Operator."""

__all__ = ["IdealOp"]

import numpy as np

from ...base import NonLinop, fft, ifft


class IdealOp(NonLinop):
    """
    Nonlinear operator for IDEAL fat-water separation.

    This operator estimates field map parameters (B0 and R2*) from multi-echo data,
    while using the fat basis computed from echo times and field strength.
    The operator performs phase-constrained linear least squares estimation of
    x = [water, fat] and computes a residual r = W*A*x*exp(i*phi)-b.
    B0 and R2* are kept as separate (real) parameters.

    Parameters
    ----------
    te : NDArray[float]
        Echo times (s), shape (ne,).
    field_strength : float
        MRI field strength (Tesla).
    b : NDArray[complex]
        Measured echo data. Shape can be (nvoxels, ne) or (nz, ny, nx, ne).
    filter_size : int, optional
        Size of the smoothing filter in k-space (default is 3).
    smooth_phase : bool, optional
        If True, smooth the initial phase using k-space filtering (default is False).

    """

    def __init__(
        self,
        te,
        field_strength,
        b,
        filter_size=3,
        smooth_phase=False,
    ):
        super().__init__()
        self.te = np.asarray(te)  # shape (ne,)
        self.field_strength = field_strength
        self.filter_size = filter_size
        self.smooth_phase = smooth_phase

        # Define the low-pass filter (3x3 ones)
        filter_kernel = np.ones((3, 3))
        self.filter_fft = fft(
            filter_kernel, shape=b.shape[1:]
        )  # Zero-padding filter to match p

        # b: if multidimensional, flatten spatial dims.
        if b.ndim > 2:
            ne = b.shape[-1]
            self.spatial_shape = b.shape[:-1]
            self.b = b.reshape(-1, ne).T  # shape (ne, nvoxels)
        else:
            self.b = b
            self.spatial_shape = None

        # Compute the fat basis A from te and field strength; A has shape (ne, 2)
        self.A = self.fat_basis(self.te, field_strength)

    def fat_basis(self, te, Tesla):
        """
        Compute the fat spectral basis for a given field strength.

        Parameters
        ----------
        te : np.ndarray
            Echo times (s).
        Tesla : float
            Field strength (T).

        Returns
        -------
        np.ndarray
            Fat basis matrix of shape (len(te), 2), where the first column is for water
            (all ones) and the second for fat (complex signal).
        """
        fat_peaks_ppm = np.array([0.9, 1.3, 1.6, 2.1, 2.75, 4.2])
        fat_amplitudes = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
        gamma_Hz_per_T = 42.57747892e6  # Hz/T
        shift_Hz = (fat_peaks_ppm - 4.7) * gamma_Hz_per_T * Tesla
        fat_signal = np.sum(
            fat_amplitudes[:, None] * np.exp(1j * 2 * np.pi * shift_Hz[:, None] * te),
            axis=0,
        )
        return np.stack([np.ones_like(te), fat_signal], axis=1)

    def transform(self, R2):
        """
        Simply returns B0 and R2* separately; Here R2* is rectified.

        Parameters
        ----------
        R2 : np.ndarray
            R2* decay rates (1/s), shape (nvoxels,).

        Returns
        -------
        R2, dR2 : tuple
            dR2 is a derivative adjustment (1 or sign(R2)) for nonnegativity.
        """
        R2 = np.abs(R2)
        dR2 = np.sign(R2) + (R2 == 0)

        return R2, dR2

    def kspace_smoothing(self, p):
        """
        Apply k-space low-pass filtering equivalent to a spatial box filter.

        Parameters
        ----------
        p : np.ndarray
            1D array to be smoothed (length = nvoxels).

        Returns
        -------
        np.ndarray
            Smoothed array (flattened).
        """
        # Compute FFT of both p and the filter, ensuring they have the same size
        p_fft = np.fft.fft2(p)

        # Perform element-wise multiplication in frequency domain
        result_fft = p_fft * self.filter_fft

        # Compute inverse FFT to obtain the convolved result
        result = np.fft.ifft2(
            result_fft
        ).real  # Take real part to remove numerical artifacts

        return result

    def update(self, x_new):
        """
        Compute common terms for the forward model and Jacobian given B0 and R2*.
        B0 and R2 are 1D arrays of length nvoxels.

        Returns
        -------
        np.ndarray
            x_out, the estimated water-fat amplitudes (2 x nvoxels).
        """
        B0 = x_new[..., 0]
        R2 = x_new[..., 1]

        self.B0 = B0
        self.R2, self.dR2 = self.transform(R2)

        # Compute complex field map W: shape (ne, nvoxels)
        self.W = np.exp(1j * self.te[:, None] * self.B0[None, :])

        # Compute WW = |W|^2, shape (ne, nvoxels)
        WW = (self.W.real) ** 2 + (self.W.imag) ** 2

        # Compute M1, M2, M3 as in MATLAB:
        # For each voxel, sum over echo times (elementwise multiplication then sum)
        A0 = (self.A[:, 0].conj() * self.A[:, 0]).real  # shape (ne,)
        A1 = (self.A[:, 0].conj() * self.A[:, 1]).real  # shape (ne,)
        A2 = (self.A[:, 1].conj() * self.A[:, 1]).real  # shape (ne,)
        self.M1 = (A0[:, None] * WW).sum(axis=0)  # shape (nvoxels,)
        self.M2 = (A1[:, None] * WW).sum(axis=0)
        self.M3 = (A2[:, None] * WW).sum(axis=0)
        self.dtm = self.M1 * self.M3 - self.M2**2  # shape (nvoxels,)

        # Compute z = inv(M) * A' * W' * b
        self.Wb = self.W.conj() * self.b  # shape (ne, nvoxels)
        z_tmp = self.A.conj().T @ self.Wb  # shape (2, nvoxels)
        z_tmp /= self.dtm  # broadcast division along rows
        self.z = 0.0 * z_tmp
        self.z[0, :] = self.M3 * z_tmp[0, :] - self.M2 * z_tmp[1, :]
        self.z[1, :] = self.M1 * z_tmp[1, :] - self.M2 * z_tmp[0, :]

        # Compute p = z' * M * z for each voxel
        self.p = (
            self.z[0, :] * self.M1 * self.z[0, :]
            + self.z[1, :] * self.M3 * self.z[1, :]
            + self.z[0, :] * self.M2 * self.z[1, :]
            + self.z[1, :] * self.M2 * self.z[0, :]
        )

        # Smooth the phase if required
        if self.smooth_phase:
            self.p = self.kspace_smoothing(self.p)

        self.phi = (
            np.angle(self.p) / 2
        )  # initial phase estimate (nvoxels,); -pi/2 < phi0 < pi/2
        self.x_out = (self.z * np.exp(-1j * self.phi)).real  # estimated amplitudes

        # Absorb sign of x into phi
        self.x_out = self.x_out * np.exp(1j * self.phi)
        self.phi = np.angle((self.x_out).sum(axis=0))  # combine components
        self.x_out = (self.z * np.exp(-1j * self.phi)).real

        # box constraint (0<=FF<=1)
        self.x_out = np.maximum(self.x_out, 0)
        self.WAx = self.W * (self.A @ self.x_out)  # shape (ne, nvoxels)
        a = self.WAx @ self.b / max((self.WAx**2).sum(), np.finfo(float).eps)
        self.x_out *= np.abs(a)
        self.phi = np.angle(a)
        if self.smooth_phase:
            self.p = self.kspace_smoothing(self.p)
            self.phi = np.angle(self.p)

        super().update(x_new)

    def _compute_forward(self, x_new):
        """
        Compute the forward model F(x) for given B0 and R2*.

        Returns the residual r = W*A*x*exp(i*phi) - b.
        B0 and R2 are 1D arrays of length nvoxels.
        """
        eiphi = np.exp(1j * self.phi)  # shape (nvoxels,)

        # Compute WAx = W * (A @ x_out)
        self.WAx = self.W * (self.A @ self.x_out)  # shape (ne, nvoxels)
        self.r = self.WAx * eiphi[None, :] - self.b  # shape (ne, nvoxels)

        # prepare output
        return self.r

    def _compute_jacobian(self, x_new):
        """
        Compute the Jacobian of F(x) with respect to B0 and R2*.

        Returns:
            JB, JR : dense Jacobian matrices (placeholders).
        """
        # Compute y
        y = (self.A * self.te).T @ self.Wb / self.dtm
        y = np.vstack(
            [
                self.M3 * y[0, :] - self.M2 * y[1, :],
                self.M1 * y[1, :] - self.M2 * y[0, :],
            ]
        )

        # Compute q
        q = (
            y[0, :] * self.M1 * self.z[0, :]
            + y[1, :] * self.M3 * self.z[1, :]
            + y[0, :] * self.M2 * self.z[1, :]
            + y[1, :] * self.M2 * self.z[0, :]
        )

        # Compute H
        self.WW *= self.te  # Equivalent to bsxfun(@times, te, WW)
        H1 = (self.A[:, 0].conj() * self.A[:, 0]).real.T @ self.WW
        H2 = (self.A[:, 0].conj() * self.A[:, 1]).real.T @ self.WW
        H3 = (self.A[:, 1].conj() * self.A[:, 1]).real.T @ self.WW

        # Compute s
        s = (
            self.z[0, :] * H1 * self.z[0, :]
            + self.z[1, :] * H3 * self.z[1, :]
            + self.z[0, :] * H2 * self.z[1, :]
            + self.z[1, :] * H2 * self.z[0, :]
        )

        # Compute JB
        JB = 1j * self.te * self.WAx
        dphi = -(q / self.p).real
        dphi[self.p == 0] = 0
        JB += 1j * dphi * self.WAx
        dx = y + self.z * dphi
        dx = (dx / self.eiphi).imag
        JB += self.W * (self.A @ dx)
        JB *= self.eiphi

        # Compute JR
        JR = -self.te * self.WAx
        dphi = -(q / self.p).imag + (s / self.p).imag
        dphi[self.p == 0] = 0
        JR += 1j * dphi * self.WAx
        dx = y + self.z * 1j * dphi
        dx = (dx / (-self.eiphi)).real

        Hx = np.vstack(
            [
                H1 * self.x[0, :] + H2 * self.x[1, :],
                H3 * self.x[1, :] + H2 * self.x[0, :],
            ]
        )
        Hx *= 2 / self.dtm
        dx += np.vstack(
            [
                self.M3 * Hx[0, :] - self.M2 * Hx[1, :],
                self.M1 * Hx[1, :] - self.M2 * Hx[0, :],
            ]
        )
        JR += self.W * (self.opts.A @ dx)
        JR *= self.eiphi
        JR *= self.dR2

        # Reshape return values
        self.x = self.x.reshape(2, self.nx, self.ny, self.nz)
        JB = JB.reshape(self.ne, self.nx, self.ny, self.nz)
        JR = JR.reshape(self.ne, self.nx, self.ny, self.nz)
        self.phi = self.phi.reshape(1, self.nx, self.ny, self.nz)

        return JB, JR
