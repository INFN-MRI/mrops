import numpy as np
import sigpy as sp
from scipy.fft import fftn, ifftn


class IDEAL_Operator(NonLinop):
    """
    Nonlinear operator for IDEAL fat-water separation.

    This operator estimates field map parameters (B0 and R2*) from multi-echo data,
    while using the fat basis computed from echo times and field strength.
    The operator performs phase-constrained linear least squares estimation of
    x = [water, fat] and computes a residual r = W*A*x*exp(i*phi)-b.
    B0 and R2* are kept as separate (real) parameters.

    Parameters
    ----------
    te : np.ndarray
        Echo times (s), shape (ne,).
    field_strength : float
        MRI field strength (Tesla).
    b : np.ndarray
        Measured echo data. Shape can be (ne, nvoxels) or (ne, nx, ny, nz).
    filter_size : int, optional
        Size of the smoothing filter in k-space (default is 3).
    nonnegFF : bool, optional
        Enforce non-negativity on FF (default is False).
    smooth_phase : bool, optional
        If True, smooth the initial phase using k-space filtering (default is False).
    noise : float, optional
        Noise level (default is 1e-6).

    """

    def __init__(
        self,
        te,
        field_strength,
        b,
        filter_size=3,
        nonnegFF=False,
        smooth_phase=False,
        noise=1e-6,
    ):
        super().__init__()
        self.te = np.asarray(te)  # shape (ne,)
        self.field_strength = field_strength
        self.filter_size = filter_size
        self.nonnegFF = nonnegFF
        self.smooth_phase = smooth_phase
        self.noise = noise

        # b: if multidimensional, flatten spatial dims.
        if b.ndim > 2:
            ne = b.shape[0]
            self.spatial_shape = b.shape[1:]
            self.b = b.reshape(ne, -1)  # shape (ne, nvoxels)
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
        Simply returns B0 and R2* separately; if nonnegFF is enabled, R2 is rectified.

        Parameters
        ----------
        R2 : np.ndarray
            R2* decay rates (1/s), shape (nvoxels,).

        Returns
        -------
        B0, R2, dR2 : tuple
            dR2 is a derivative adjustment (1 or sign(R2)) for nonnegativity.
        """
        if self.nonnegFF:
            R2 = np.abs(R2)
            dR2 = np.sign(R2) + (R2 == 0)
        else:
            dR2 = 1

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
        if self.spatial_shape is None:
            return p
        p_img = p.reshape(self.spatial_shape)

        # Create a box filter kernel of ones.
        kernel = np.ones(self.spatial_shape)

        # For a simple low-pass effect, use a small cube in k-space.
        # Here we simulate a 3x3x3 box filter in spatial domain by applying FFT, multiplying by a mask, and inverse FFT.
        Pk = fftn(p_img)
        H = np.zeros(self.spatial_shape)
        center = tuple(s // 2 for s in self.spatial_shape)
        slices = tuple(
            slice(c - self.filter_size // 2, c + self.filter_size // 2 + 1)
            for c in center
        )
        H[slices] = 1
        H /= H.sum()
        p_smoothed = np.real(ifftn(Pk * H))
        return p_smoothed.ravel()

    def update(self, B0, R2):
        """
        Compute common terms for the forward model and Jacobian given B0 and R2*.
        B0 and R2 are 1D arrays of length nvoxels.

        Returns
        -------
        np.ndarray
            x_out, the estimated water-fat amplitudes (2 x nvoxels).
        """
        self.B0 = B0
        self.R2, self.dR2 = self.transform(R2)

        # Compute complex field map W: shape (ne, nvoxels)
        self.W = np.exp(1j * self.te[:, None] * self.B0[None, :])

        # Compute WW = |W|^2, shape (ne, nvoxels)
        WW = np.real(self.W) ** 2 + np.imag(self.W) ** 2

        # Compute M1, M2, M3 as in MATLAB:
        # For each voxel, sum over echo times (elementwise multiplication then sum)
        A0 = np.real(self.A[:, 0].conj() * self.A[:, 0])  # shape (ne,)
        A1 = np.real(self.A[:, 0].conj() * self.A[:, 1])  # shape (ne,)
        A2 = np.real(self.A[:, 1].conj() * self.A[:, 1])  # shape (ne,)
        self.M1 = np.sum(A0[:, None] * WW, axis=0)  # shape (nvoxels,)
        self.M2 = np.sum(A1[:, None] * WW, axis=0)
        self.M3 = np.sum(A2[:, None] * WW, axis=0)
        self.dtm = self.M1 * self.M3 - self.M2**2  # shape (nvoxels,)

        # Compute z = inv(M) * A' * W' * b
        self.Wb = self.W.conj() * self.b  # shape (ne, nvoxels)
        z_tmp = self.A.conj().T @ self.Wb  # shape (2, nvoxels)
        z_tmp /= self.dtm  # broadcast division along rows
        self.z = np.empty_like(z_tmp)
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
        self.x_out = np.real(self.z * np.exp(-1j * self.phi))  # estimated amplitudes

        # Absorb sign of x into phi
        self.x_out = self.x_out * np.exp(1j * self.phi)
        self.phi = np.angle(np.sum(self.x_out, axis=0))  # combine components
        self.x_out = np.real(self.z * np.exp(-1j * self.phi))

        # box constraint (0<=FF<=1)
        if self.nonnegFF:
            self.x_out = np.maximum(self.x_out, 0)
            self.WAx = self.W * (self.A @ self.x_out)  # shape (ne, nvoxels)
            a = self.WAx @ self.b / max((self.WAx**2).sum(), np.finfo(float).eps)
            self.x_out *= np.abs(a)
            self.phi = np.angle(a)
            if self.smooth_phase:
                self.p = self.kspace_smoothing(self.p)
                self.phi = np.angle(self.p)

        return self.x_out

    def _compute_forward(self, B0, R2):
        """
        Compute the forward model F(x) for given B0 and R2*.

        Returns the residual r = W*A*x*exp(i*phi) - b.
        B0 and R2 are 1D arrays of length nvoxels.
        """
        eiphi = np.exp(1j * self.phi)  # shape (nvoxels,)

        # Compute WAx = W * (A @ x_out)
        self.WAx = self.W * (self.A @ self.x_out)  # shape (ne, nvoxels)
        self.r = self.WAx * eiphi[None, :] - self.b  # shape (ne, nvoxels)
        if self.spatial_shape is not None:
            ne = self.te.shape[0]
            self.r = self.r.reshape((ne,) + self.spatial_shape)

        return self.r

    def _compute_jacobian(self, B0, R2):
        """
        Compute the Jacobian of F(x) with respect to B0 and R2*.

        Returns:
            JB, JR : dense Jacobian matrices (placeholders).
        """
        # Compute real part (B0) of the Jacobian (JB)
        JB = 1j * self.te[:, None] * self.WAx
        dphi = -np.real(self.q / self.p)
        dphi[self.p == 0] = 0
        JB += self.WAx * 1j * dphi
        dx = self.y + self.z * dphi
        dx = np.imag(dx / np.exp(1j * self.phi))
        JB += self.W * (self.A @ dx)
        JB *= np.exp(1j * self.phi)  # Apply phase

        # Compute imaginary part (R2*) of the Jacobian (JR)
        JR = -self.te[:, None] * self.WAx
        dphi = -np.imag(self.q / self.p) + np.imag(self.s / self.p)
        dphi[self.p == 0] = 0
        JR += self.WAx * 1j * dphi
        dx = self.y + self.z * 1j * dphi
        dx = np.real(dx / np.exp(1j * self.phi))
        Hx = self.H1 * self.x_out[0, :] + self.H2 * self.x_out[1, :]
        Hx = np.stack([self.H3 * self.x_out[1, :] + self.H2 * self.x_out[0, :]])
        Hx = Hx * 2 / self.dtm
        dx += self.M3 * Hx[0, :] - self.M2 * Hx[1, :]
        dx += self.M1 * Hx[1, :] - self.M2 * Hx[0, :]
        JR += self.W * (self.A @ dx)
        JR *= np.exp(1j * self.phi)
        JR *= self.dR2

        return JB, JR
