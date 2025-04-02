"""IDEAL nonlinear optimization of fieldmap."""

__all__ = []

from types import SimpleNamespace

import numpy as np

from numpy.typing import NDArray

from ..._sigpy import get_device

from ...base import NonLinop
from ...optimize import IrgnmCauchy

from ._ideal_op import IdealOp


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
    muR : float, optional
        Regularization strength for B0 part of the complex field map.
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
        (if ``ret_residual == False`` and ``finalize == False``).
    r : ND
        Residuals after fitting of shape ``(nte, *matrix_size)`` (if ``ret_residual == True``).
    B0 : NDArray[float]
        B0 field map in ``[rad/s]`` of shape ``(*matrix_size)``
        (if ``ret_residual == False`` and ``finalize == True``).
    R2* : NDArray[float]
        R2* field map in ``[Hz]`` of shape ``(*matrix_size)``
        (if ``ret_residual == False`` and ``finalize == True``).
    phi0 : NDArray[float]
        Background field map in ``[rad]`` of shape ``(*matrix_size)``
        (if ``ret_residual == False`` and ``finalize == True``).
    FF : NDArray[float]
        Fat fraction map of shape ``(*matrix_size)``
        (if ``ret_residual == False`` and ``finalize == True``).
    WF : NDArray[float]
        Water fraction map``[Hz]`` of shape ``(*matrix_size)``
        (if ``ret_residual == False`` and ``finalize == True``).

    """
    if finalize is True and ret_residual is True:
        raise ValueError("If finalize == True, ret_residual must be False")
    if psi0 is None:
        psi = linear_fieldmap(echo_series, te, fat_model.chemshift)
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

    # Return residual only
    if ret_residual:
        return ideal_solver.residual

    # Final output
    if finalize:
        phi, ff, wf = ideal_solver.extras
        return psi.real / 2 / np.pi, abs(psi.imag), phi, ff, wf

    # Return updated complex psi
    return psi


def linear_fieldmap(
    echo_series: NDArray[complex],
    te: NDArray[float],
    chemshift: float,
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
    chemshift : float
        Main fat frequency offset in ``[Hz]``.
    finalize : bool, optional
        If ``True``, returns field maps. The default is ``False``.

    Returns
    -------
    psi : NDArray[complex]
        Complex field map ``B0 + 1j * R2*`` of shape ``(*matrix_size)``
        (if ``finalize == False``).
    B0 : NDArray[float]
        B0 field map in ``[rad/s]`` of shape ``(*matrix_size)``
        (if ``ret_residual == False`` and ``finalize == True``).
    R2* : NDArray[float]
        R2* field map in ``[Hz]`` of shape ``(*matrix_size)``
        (if ``ret_residual == False`` and ``finalize == True``).

    """
    xp = get_device(echo_series).xp
    ne = echo_series.shape[0]  # TE is on the first axis

    # Compute dot product along the TE axis
    tmp = xp.tensordot(echo_series[: ne - 1], echo_series[1:ne], axes=([0], [0]))
    psi = xp.angle(tmp) / np.min(np.diff(te)) + 1j * xp.imag(chemshift)

    # Compute another version using a limited number of TE steps
    othertmp = xp.tensordot(
        echo_series[: min(ne - 1, 3)], echo_series[1 : min(ne, 4)], axes=([0], [0])
    )
    dte = xp.diff(te)
    psi = xp.angle(othertmp) / xp.min(dte[: min(ne - 1, 4)]) + 1j * xp.imag(chemshift)

    # Final output
    if finalize:
        return psi.real / 2 / np.pi, abs(psi.imag)

    # Return initial complex psi
    return psi


# %% utils
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
        self.A.update(self.x)
        return self.A.forward() - self.b

    @property
    def extras(self):
        xp = get_device(self.x).xp
        self.A.update(self.x)

        # get variables
        phi = self.A.phi
        x = self.A.x
        x = x / x.sum(axis=0)  # normalize to 1
        x = xp.nan_to_num(x, posinf=0.0, neginf=0.0)

        # unpack water fraction and fat fraction
        ff = x[1]
        wf = x[0]

        return phi, ff, wf
