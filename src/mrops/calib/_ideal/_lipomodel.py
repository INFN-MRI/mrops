"""Fat multipeak model."""

__all__ = ["lipomodel"]

import warnings

from types import SimpleNamespace

from numpy.typing import NDArray

import numpy as np
from scipy.optimize import minimize

from mrinufft._array_compat import with_numpy


def lipomodel(te: NDArray[float], field_strength: float) -> SimpleNamespace:
    """
    Computes the fat-water matrix and estimates the main chemical shift.

    Parameters
    ----------
    te : NDArray[float]
        Echo times in seconds.
    Tesla : float
        Field strength of the MRI scanner.

    Returns
    -------
    basis : NDArray[float]
        Matrix containing [water fat] basis vectors.
    chemshift : complex
        Best fit of fat to `exp(i * B0 * te - R2 * te)`,
        where `B0 = real(psif)` (rad/s) and `R2 = imag(psif)`.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        basis, chemshift = _lipomodel(te, field_strength)

    return SimpleNamespace(basis=basis, chemshift=chemshift)


# %% utils
@with_numpy
def _lipomodel(te, field_strength):
    if np.max(te) < 1e-3 or np.max(te) > 1:
        raise ValueError("'te' should be in seconds.")

    # Ensure te is a column vector
    te = np.asarray(te).reshape(-1, 1)

    # Hardcoded opt values
    gyro = 42.57747892  # Gyromagnetic ratio in MHz/T
    species = {
        "water": {"frequency": 0, "relAmps": 1},
        "fat": {
            "frequency": np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]),  # ppm
            "relAmps": np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048]),
        },
    }

    # Larmor frequency (MHz)
    larmor = gyro * field_strength

    # Extract water and fat properties
    H2O = species["water"]["frequency"]  # Water frequency in ppm
    water = species["water"]["relAmps"] * np.ones_like(te)

    ampl = species["fat"]["relAmps"]
    freq = larmor * (species["fat"]["frequency"] - H2O)  # Fat peak frequencies in Hz

    # Compute fat signal
    FatA = ampl * np.exp(2j * np.pi * te * freq)
    fat = np.sum(FatA, axis=1, keepdims=True)

    # Construct fat-water matrix
    basis = np.hstack((water, fat))

    # Nonlinear fitting of fat to complex exponential
    def myfun(psif, te, data):
        """Residual function for optimization."""
        psif = complex(psif[0], psif[1])
        f = np.exp(1j * psif * te)
        v = np.vdot(f, data) / np.vdot(f, f)
        r = v * f - data
        return np.linalg.norm(r)

    # Initial estimate for psif (B0 and R2 in rad/s)
    chemshift_init = np.array([2 * np.pi * larmor * (1.3 - H2O), 50])

    # Optimize psif using scipy's minimize (equivalent to MATLAB's fminsearch)
    res = minimize(myfun, chemshift_init, args=(te, fat), method="Nelder-Mead")
    chemshift = complex(res.x[0], res.x[1])

    return basis, chemshift
