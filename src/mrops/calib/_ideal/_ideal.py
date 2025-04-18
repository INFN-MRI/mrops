"""IDEAL nonlinear optimization of fieldmap."""

__all__ = ["prisco_calib", "fieldmap"]

import math
from types import SimpleNamespace

import numpy as np

from numpy.typing import NDArray

from mrinufft._array_compat import with_numpy_cupy

from ..._sigpy import get_device, to_device

from ...base import NonLinop
from ...optimize import IrgnmCauchy

from ._lipomodel import lipomodel
from ._ideal_op import IdealOp
from ._unswap import Unswap


@with_numpy_cupy
def prisco_calib(
    echo_series: NDArray[complex],
    te: NDArray[float],
    field_strength: float,
    muB: float = 0.03,
    muR: float = 0.1,
    max_iter: int = 10,
    irgnm_iter: int = 10,
    linesearch_iter: int = 5,
    smoothfilt_width: int = 64,
    medfilt_size: int = 3,
):
    """
    Nonlinear estimation of complex field map and fat fraction.

    Parameters
    ----------
    echo_series : NDArray[complex]
        Measured data in image space of shape ``(nte, *matrix_size)``.
    te : NDArray[float]
        Echo times in ``[ms]`` of shape ``(nte,)``.
    field_strength : float
        Static field strength in ``[T]``.
    muB : float, optional
        Regularization strength for B0 part of the complex field map.
        The default is ``0.03``.
    muR : float, optional
        Regularization strength for R2* part of the complex field map.
        The default is ``0.1``.
    max_iter : int, optional
        Number of IRGN steps. The default is ``10``.
    irgnm_iter : int, optional
        Number of IRGN iterations per step. The default is ``10``.
    linesearch_iter : int, optional
        Number of linesearch iterations for each IRGN iteration. The default is ``5``.
    smoothfilt_width : int | list[int] | tuple[int], optional
        Size of low frequency region. If scalar, assumes isotropic.
        The default is ``64``
    medfilt_size : int, optional
        Size of median filter. The default is ``3``.


    Returns
    -------
    B0 : NDArray[float]
        B0 map in ``[Hz]`` of shape ``(*matrix_size)``.
    R2s : NDArray[float]
        Rectified R2* map in ``[Hz]`` of shape ``(*matrix_size)``.
    phi0 : NDArray[float]
        Background field map in ``[rad]`` of shape ``(*matrix_size)``
        (if ``finalize == True``).
    FF : NDArray[float]
        Fat fraction map of shape ``(*matrix_size)``
        (if ``finalize == True``).
    WF : NDArray[float]
        Water fraction map``[Hz]`` of shape ``(*matrix_size)``
        (if ``finalize == True``).

    """
    # get device and backend
    device = get_device(echo_series)
    xp = device.xp

    # rescale and convert units
    te = te * 1e-3  # [ms] -> [s]
    echo_series = echo_series / xp.quantile(abs(echo_series[0]), 0.95)

    # estimate noise
    noise = estimate_noise_std(echo_series)

    # build mask
    mask = (abs(echo_series) ** 2).sum(axis=0) ** 0.5 > 10 * noise

    # rescale regularization
    muB *= noise
    muR *= noise

    # get fat model
    fat_model = lipomodel(te, field_strength)
    fat_model.basis = to_device(fat_model.basis, device)

    # get unswap operator
    unswap = Unswap(
        te,
        fat_model.chemshift.real,
        mask,
        medfilt_size,
        smoothfilt_width,
    )

    # first iteration
    r, psi = nonlinear_fieldmap(
        echo_series,
        te,
        fat_model,
        0.0,
        muR,
        None,
        irgnm_iter,
        linesearch_iter,
        ret_residual=True,
    )
    s, _ = nonlinear_fieldmap(
        echo_series,
        te,
        fat_model,
        0.0,
        muR,
        psi,
        irgnm_iter,
        linesearch_iter,
        ret_residual=True,
    )
    bad = (abs(r) ** 2).sum(axis=0) > (abs(s) ** 2).sum(axis=0)
    psi[bad] -= fat_model.chemshift

    # from 1 to n-th iteration
    if max_iter > 1:
        for _ in range(1, max_iter - 1):
            psi = nonlinear_fieldmap(
                echo_series, te, fat_model, muB, muR, psi, irgnm_iter, linesearch_iter
            )
            psi = unswap(psi)

    # last iteration
    psi, phi, ff, wf = nonlinear_fieldmap(
        echo_series,
        te,
        fat_model,
        muB,
        muR,
        psi,
        irgnm_iter,
        linesearch_iter,
        finalize=True,
    )
    psi = unswap(psi)

    return psi.real / 2 / math.pi, abs(psi.imag), phi, ff, wf


def nonlinear_fieldmap(
    echo_series: NDArray[complex],
    te: NDArray[float],
    fat_model: SimpleNamespace,
    muB: float = 0.03,
    muR: float = 0.1,
    psi0: NDArray[complex] | None = None,
    max_iter: int = 10,
    linesearch_iter: int = 5,
    ret_residual: bool = False,
    finalize: bool = False,
):
    """
    Nonlinear estimation of complex field map.

    Parameters
    ----------
    echo_series : NDArray[complex]
        Measured data in image space of shape ``(nte, *matrix_size)``.
    te : NDArray[float]
        Echo times in ``[s]`` of shape ``(nte,)``.
    fat_model : SimpleNamespace
        Fat signal model containing ``basis`` and ``chemshift``.
    muB : float, optional
        Regularization strength for B0 part of the complex field map.
        The default is ``0.03``.
    muR : float, optional
        Regularization strength for R2* part of the complex field map.
        The default is ``0.1``.
    psi0 : NDArray[complex], optional
        Initial estimate of the complex field map ``B0 + 1j * R2*``
        of shape ``(*matrix_size)``. If not provided, is estimated from ``echo_series``.
    max_iter : int, optional
        Number of IRGN iterations. The default is ``10``.
    linesearch_iter : int, optional
        Number of linesearch iterations for each IRGN iteration. The default is ``5``.
    ret_residual : bool, optional
        If ``True``, returns residual between measurements and prediction.
        The default is ``False``.
    finalize : bool, optional
        If ``True``, returns field maps, background phase and water/fat fractions.
        The default is ``False``.


    Returns
    -------
    psi : NDArray[complex]
        Complex field map ``B0 + 1j * R2*`` of shape ``(*matrix_size)``
        (if ``finalize == False``).
    r : ND
        Residuals after fitting of shape ``(nte, *matrix_size)``
        (if ``finalize == False`` and ``ret_residual == True``).
    phi0 : NDArray[float]
        Background field map in ``[rad]`` of shape ``(*matrix_size)``
        (if ``finalize == True``).
    FF : NDArray[float]
        Fat fraction map of shape ``(*matrix_size)``
        (if ``finalize == True``).
    WF : NDArray[float]
        Water fraction map``[Hz]`` of shape ``(*matrix_size)``
        (if ``finalize == True``).

    """
    if ret_residual is True and finalize is True:
        raise ValueError("If ret_residual == True, finalize must be False")
    if psi0 is None:
        psi = fieldmap(echo_series, te, fat_model.chemshift)
    else:
        psi = psi0.copy()

    # Regularizer (smooth B0 + zero R2)
    psi0 = psi.real + 1j * 0.0 * psi.imag

    # Set up IDEAL operator
    ideal_op = IdealOp(echo_series, te, fat_model.basis)

    # Set up cost function for line search
    costfun = IdealCost(ideal_op, echo_series, psi0, muB, muR)

    # Set up nonlinear optimization
    weights = abs(echo_series[0])
    ideal_solver = IrgnmIdeal(
        ideal_op,
        echo_series,
        psi,
        psi0,
        muB + 1j * muR,
        costfun,
        linesearch_iter,
        max_iter,
        weights,
    )
    psi = ideal_solver.run()

    # Final output
    if finalize:
        phi, ff, wf = ideal_solver.extras
        return psi, phi, ff, wf

    # Return residual only
    if ret_residual:
        return ideal_solver.residual, psi

    # Return updated complex psi
    return psi


def fieldmap(
    echo_series: NDArray[complex],
    te: NDArray[float],
    chemshift: complex = 0.0,
    finalize: bool = False,
):
    """
    Computes the dominant frequency from 4D data.

    Parameters
    ----------
    echo_series : NDArray[complex]
        Measured data in image space of shape ``(nte, *matrix_size)``.
    te : NDArray[float]
        Echo times in ``[s]`` of shape ``(nte,)``.
    chemshift : complex
        Main fat frequency offset in ``[Hz]``. The default is ``0.0``.
    finalize : bool, optional
        If ``True``, returns field maps. The default is ``False``.

    Returns
    -------
    psi : NDArray[complex]
        Complex field map ``B0 + 1j * R2*`` of shape ``(*matrix_size)``.

    """
    xp = get_device(echo_series).xp
    ne = echo_series.shape[0]  # TE is on the first axis

    # Compute dot product along the TE axis
    tmp = (echo_series[: min(ne - 1, 3)].conj() * echo_series[1 : min(ne, 4)]).sum(
        axis=0
    )
    psi = xp.angle(tmp) / np.min(np.diff(te)) + 1j * xp.imag(chemshift)

    return psi


# %%
def estimate_noise_std(input):
    """
    Estimate the noise standard deviation from the given data.

    Parameters
    ----------
    input : NDArrat[complex | float]
        The input 2D/3D array (image or dataset) from which noise is estimated.

    Returns
    -------
    float
        Estimated noise standard deviation.

    Notes
    -----
    - The function constructs a matrix by circularly shifting the data within
      a 5×5 window, then finds the smallest singular value.
    - The smallest singular value is assumed to represent noise.
    - The estimated noise is normalized based on the number of nonzero elements.

    """
    xp = get_device(input).xp
    X_list = []
    for j in range(-2, 3):
        for k in range(-2, 3):
            shifted_data = xp.roll(input, shift=(j, k), axis=(-2, -1))
            X_list.append(shifted_data.ravel())  # Flatten and store

    X = xp.column_stack(X_list)  # Construct matrix from column vectors
    smallest_sval = xp.min(xp.linalg.svd(X.conj().T @ X, compute_uv=False))

    return (smallest_sval / xp.count_nonzero(input)) ** 0.5


class IdealCost:
    """IDEAL cost function for linesearch"""

    def __init__(
        self,
        A: NonLinop,
        b: NDArray[complex],
        x0: NDArray[complex],
        muB: float,
        muR: float,
    ):
        self.A = A  # nonlinear operator describing the model
        self.b = b  # observations
        self.x0 = x0  # initial solution for regularization
        self.muB = muB  # regularization strength for B0
        self.muR = muR  # regularization strength for R2*

    def __call__(self, input):
        self.A.update(input)  # x = psi
        rhs = self.A.forward() - self.b
        bias_re = input.real - self.x0
        bias_im = abs(input.imag) - self.x0
        return (
            (abs(rhs) ** 2).sum(axis=0)
            + (self.muB * bias_re) ** 2
            + (self.muR * bias_im) ** 2
        )


class IrgnmIdeal(IrgnmCauchy):
    """IDEAL nonlinear optimization"""

    @property
    def residual(self):
        self.alg.A.update(self.alg.x)
        return self.alg.A.forward() - self.alg.b

    @property
    def extras(self):
        xp = get_device(self.alg.x).xp
        self.alg.A.update(self.alg.x)

        # get variables
        phi = self.alg.A.phi
        x = self.alg.A.maps
        x = x / x.sum(axis=0)  # normalize to 1
        x = xp.nan_to_num(x, posinf=0.0, neginf=0.0)

        # unpack water fraction and fat fraction
        ff = x[1]
        wf = x[0]

        return phi, ff, wf
