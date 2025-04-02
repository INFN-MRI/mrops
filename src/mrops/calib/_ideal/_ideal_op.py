"""IDEAL Nonlinear Operator."""

__all__ = ["IdealOp"]

from numpy.typing import NDArray

from ..._sigpy import get_device
from ...base import NonLinop

from ._ideal_reg import nonnegative_constraint


class IdealOp(NonLinop):
    """
    Nonlinear operator for IDEAL fat-water separation.

    Given the complex fieldmap ``psi = B0 + 1j * R2*``,
    the operator performs phase-constrained linear least squares estimation of the (complex)
    ``x = [water_frac, fat_frac]`` and initial phase ``phi``.

    It also computes computes forward model F_n = W*A*x*exp(i*phi), with
    ``W = diag(exp(1j * psi * te))`` being the time evolution
    of the complex field map and ``A`` being the complex ``(nte, 2)`` basis
    describing time evolution of water and fat, as well as its jacobian with respect
    to B0 and R2*.

    Parameters
    ----------
    b : NDArray[complex]
        Measured echo data. Shape can be (nvoxels, ne) or (nz, ny, nx, ne).
    te : NDArray[float]
        Echo times (s), shape (ne,).
    field_strength : float
        MRI field strength (Tesla).

    Attributes
    ----------
    x: NDArray[complex]
        Complex water and fat fraction array of shape ``(2, *b.shape[1:])``
    phi: NDArray[float]
        Initial phase of shape ``(*b.shape[1:])``

    """

    def __init__(
        self,
        b: NDArray[complex],
        te: NDArray[float],
        field_strength: float,
        # filter_size: int = 3,
        # smooth_phase: bool = False,
    ):
        super().__init__()
        device = get_device(b)
        self._xp = device.xp
        self._shape = b.shape[1:]
        self._b = b.reshape(b.shape[0], -1)  # shape (ne, nvoxels)
        self._te = te

        # Compute the fat basis A from te and field strength; A has shape (ne, 2)
        self._A = self.fat_basis(te, field_strength)

        # Define the low-pass filter (3x3 ones)
        # self._smooth_phase = smooth_phase
        # if smooth_phase:
        #     self.kspace_smoothing = LowPassFilter(device, b.shape[1:], filter_size)

    def fat_basis(self, te, field_strength):
        """
        Compute the fat spectral basis for a given field strength.

        Parameters
        ----------
        te : np.ndarray
            Echo times (s).
        field_strength : float
            Field strength (T).

        Returns
        -------
        np.ndarray
            Fat basis matrix of shape (len(te), 2), where the first column is for water
            (all ones) and the second for fat (complex signal).
        """
        xp = self._xp

        # Multi-peak fat model
        fat_peaks_ppm = xp.array([0.9, 1.3, 1.6, 2.1, 2.75, 4.2], dtype=xp.float32)
        fat_amplitudes = xp.array(
            [0.087, 0.693, 0.128, 0.004, 0.039, 0.048], dtype=xp.float32
        )

        # Compute frequency shift for each peak
        gamma_Hz_per_T = 42.57747892e6  # Hz/T
        shift_Hz = (fat_peaks_ppm - 4.7) * gamma_Hz_per_T * field_strength

        # Compute basis
        fat_signal = xp.sum(
            fat_amplitudes[:, None] * xp.exp(1j * 2 * xp.pi * shift_Hz[:, None] * te),
            axis=0,
        )
        return xp.stack([0 * te + 1, fat_signal], axis=1)

    @property
    def phi(self):
        return self._phi.reshape(*self._shape)

    @property
    def x(self):
        return self._x.reshape(-1, *self._shape)

    def update(self, psi):
        """
        Performs update of fat, water, phi0, B0, R2* and forward model.

        This is accomplished by phase constrained least squares optimization.

        Since the problem is very small, least squares is performed by manual
        inversion of the (2, 2) system matrix (i.e., computing determinant).

        """
        xp = self._xp
        B0 = psi.real.copy()
        R2 = psi.imag.copy()

        self._B0 = B0
        self._R2, self._dR2 = nonnegative_constraint(R2)

        # Compute complex field map W: shape (ne, nvoxels)
        self._W = xp.exp(1j * self._te[:, None] * self._B0[None, :])

        # Compute WW = |W|^2, shape (ne, nvoxels)
        WW = (self._W.real) ** 2 + (self._W.imag) ** 2

        # Compute M1, M2, M3 as in MATLAB:
        # For each voxel, sum over echo times (elementwise multiplication then sum)
        A0 = (self._A[:, 0].conj() * self._A[:, 0]).real  # shape (ne,)
        A1 = (self._A[:, 0].conj() * self._A[:, 1]).real  # shape (ne,)
        A2 = (self._A[:, 1].conj() * self._A[:, 1]).real  # shape (ne,)
        self._M1 = (A0[:, None] * WW).sum(axis=0)  # shape (nvoxels,)
        self._M2 = (A1[:, None] * WW).sum(axis=0)
        self._M3 = (A2[:, None] * WW).sum(axis=0)
        self._dtm = self._M1 * self._M3 - self._M2**2  # shape (nvoxels,)

        # Compute z = inv(M) * A' * W' * b
        self._Wb = self._W.conj() * self._b  # shape (ne, nvoxels)
        z_tmp = self._A.conj().T @ self._Wb  # shape (2, nvoxels)
        z_tmp /= self.dtm  # broadcast division along rows
        self._z = 0.0 * z_tmp
        self._z[0, :] = self._M3 * z_tmp[0, :] - self._M2 * z_tmp[1, :]
        self._z[1, :] = self._M1 * z_tmp[1, :] - self._M2 * z_tmp[0, :]

        # Compute p = z' * M * z for each voxel
        self._p = (
            self._z[0, :] * self._M1 * self._z[0, :]
            + self._z[1, :] * self._M3 * self._z[1, :]
            + self._z[0, :] * self._M2 * self._z[1, :]
            + self._z[1, :] * self._M2 * self._z[0, :]
        )

        # Smooth the phase if required
        # if self._smooth_phase:
        #     self._p = self.kspace_smoothing(self._p)

        self._phi = (
            xp.angle(self._p) / 2
        )  # initial phase estimate (nvoxels,); -pi/2 < phi0 < pi/2
        self._x = (self._z * xp.exp(-1j * self._phi)).real  # estimated amplitudes

        # Absorb sign of x into phi
        self._x = self._x * xp.exp(1j * self._phi)
        self._phi = xp.angle((self._x).sum(axis=0))  # combine components
        self._x = (self._z * xp.exp(-1j * self._phi)).real

        # box constraint (0<=FF<=1)
        self._x = xp.maximum(self._x, 0)
        self._WAx = self.W * (self._A @ self._x)  # shape (ne, nvoxels)
        a = self._WAx @ self._b / max((self._WAx**2).sum(), xp.finfo(float).eps)
        self._x *= xp.abs(a)
        self._phi = xp.angle(a)
        # if self._smooth_phase:
        #     self._p = self.kspace_smoothing(self._p)
        #     self._phi = xp.angle(self._p)

        # Compute WAx = W * (A @ x_out)
        self._eiphi = xp.exp(1j * self._phi)  # shape (nvoxels,)
        self._WAx = self._W * (self._A @ self._x)  # shape (ne, nvoxels)

        # Store forward model
        self.F_n = (self._WAx * self._eiphi[None, :]).reshape(-1, *self._shape)

    def jacobian(self):
        """
        Return the current Jacobian operator dF(x).

        Jacobian is computed with respect to B0 (``self.jecobian()[0]``)
        and R2* (``self.jecobian()[1]``).

        Returns
        -------
        NDArray[comples]
            The ``(2, *b.shape[1:])`` Jacobian operator evaluated at the last update point.

        """
        self._compute_jacobian()
        return self.DF_n

    def _compute_jacobian(self):
        """Compute the Jacobian of F(x) with respect to B0 and R2*."""
        xp = self._xp
        y = (self._A * self._te).T @ self._Wb / self._dtm
        y = xp.vstack(
            [
                self._M3 * y[0, :] - self._M2 * y[1, :],
                self._M1 * y[1, :] - self._M2 * y[0, :],
            ]
        )

        # Compute q
        q = (
            y[0, :] * self._M1 * self._z[0, :]
            + y[1, :] * self._M3 * self._z[1, :]
            + y[0, :] * self._M2 * self._z[1, :]
            + y[1, :] * self._M2 * self._z[0, :]
        )

        # Compute H
        self._WW *= self._te  # Equivalent to bsxfun(@times, te, WW)
        H1 = (self._A[:, 0].conj() * self._A[:, 0]).real.T @ self._WW
        H2 = (self._A[:, 0].conj() * self._A[:, 1]).real.T @ self._WW
        H3 = (self._A[:, 1].conj() * self._A[:, 1]).real.T @ self._WW

        # Compute s
        s = (
            self._z[0, :] * H1 * self._z[0, :]
            + self._z[1, :] * H3 * self._z[1, :]
            + self._z[0, :] * H2 * self._z[1, :]
            + self._z[1, :] * H2 * self._z[0, :]
        )

        # Compute JB
        JB = 1j * self._te * self._WAx
        dphi = -(q / self._p).real
        dphi[self._p == 0] = 0
        JB += 1j * dphi * self._WAx
        dx = y + self._z * dphi
        dx = (dx / self._eiphi).imag
        JB += self._W * (self._A @ dx)
        JB *= self._eiphi

        # Compute JR
        JR = -self._te * self._WAx
        dphi = -(q / self._p).imag + (s / self._p).imag
        dphi[self._p == 0] = 0
        JR += 1j * dphi * self._WAx
        dx = y + self._z * 1j * dphi
        dx = (dx / (-self._eiphi)).real

        Hx = xp.vstack(
            [
                H1 * self._x[0, :] + H2 * self._x[1, :],
                H3 * self._x[1, :] + H2 * self._x[0, :],
            ]
        )
        Hx *= 2 / self.dtm
        dx += xp.vstack(
            [
                self._M3 * Hx[0, :] - self._M2 * Hx[1, :],
                self._M1 * Hx[1, :] - self._M2 * Hx[0, :],
            ]
        )
        JR += self._W * (self._A @ dx)
        JR *= self._eiphi
        JR *= self._dR2

        # Reshape return values
        JB = JB.reshape(-1, *self._shape)
        JR = JR.reshape(-1, *self._shape)
        self.DF_n = xp.stack((JB, JR), axis=0)
