"""IDEAL nonlinear optimization of fieldmap."""

__all__ = []

import gc
from typing import Callable

import numpy as np

from numpy.typing import NDArray

from mrinufft._array_compat import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp

from ..._sigpy import get_device
from ..._sigpy.alg import Alg
from ..._sigpy.app import App

from ...base import NonLinop

from ._ideal_reg import WeightedMean


def nlsfit(psi, te, data, opts):
    """
    Nonlinear Least Squares Fitting

    Parameters:
        psi : numpy array (complex)
            Initial estimate of the parameter to be optimized.
        te : numpy array
            Echo times or independent variable in the fitting.
        data : numpy array
            Observed data to be fitted.
        opts : dict
            Dictionary containing options such as:
                - 'maxit' : list (number of iterations)
                - 'muB' : float (regularization for B0 smoothness)
                - 'muR' : float (regularization for R2 zero)
                - 'mask' : boolean mask array for valid data points

    Returns:
        r : numpy array
            Residuals after fitting.
        psi : numpy array
            Optimized parameter.
        phi : numpy array
            Phase component.
        x : numpy array
            Additional output from pclsr.
    """

    # Regularizer (smooth B0 + zero R2)
    PSI = np.real(psi)

    for _ in range(opts["maxit"][0]):  # Outer iteration loop
        # Compute residual and Jacobian
        r, phi, x, JB, JR = pclsr(psi, te, data, opts)

        # Compute gradient G = J^T * r
        gB = np.real(np.dot(JB, r)) + opts["muB"] ** 2 * np.real(psi - PSI)
        gR = np.real(np.dot(JR, r)) + opts["muR"] ** 2 * np.imag(psi - PSI)
        G = gB + 1j * gR  # Complex gradient

        # Compute approximate Hessian
        H1 = np.real(np.dot(JB, JB)) + opts["muB"] ** 2
        H2 = np.real(np.dot(JB, JR))
        H3 = np.real(np.dot(JR, JR)) + opts["muR"] ** 2

        # Compute Cauchy step size
        GG = gB**2 + gR**2
        GHG = gB**2 * H1 + 2 * gB * gR * H2 + gR**2 * H3

        # Stabilize step size
        damp = np.median(GHG[opts["mask"]]) / 1000
        step = GG / (GHG + damp) / opts["maxit"][2]
        dpsi = -step * G

        # Define cost function
        def cost(arg):
            return (
                np.sum(np.abs(pclsr(arg, te, data, opts)) ** 2)
                + opts["muB"] ** 2 * np.sum(np.real(transform(arg, opts) - PSI) ** 2)
                + opts["muR"] ** 2 * np.sum(np.imag(transform(arg, opts) - PSI) ** 2)
            )

        # Basic line search
        for _ in range(opts["maxit"][2]):
            if cost(psi + dpsi) < cost(psi):
                psi += dpsi
            else:
                dpsi /= 10

    return r, psi, phi, x


class _IDEALCost:
    """
    IDEAL cost function for linesearch
    """

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


# %% utils
class LineSearch(App):
    """Basic linesearch algorithm."""

    def __init__(
        self,
        cost: Callable,
        x: NDArray[complex],
        dx: NDArray[complex],
        x0: NDArray[complex],
        max_iter: int = 10,
    ):
        _alg = _LineSearch(
            cost,
            x,
            dx,
            x0,
            max_iter,
        )
        super().__init__(_alg, False, False, False)

    def _output(self):
        gc.collect()
        if CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()
        return self.alg.x


class _LineSearch(Alg):
    """Basic linesearch algorithm."""

    def __init__(
        self,
        cost: Callable,
        x: NDArray[complex],
        dx: NDArray[complex],
        x0: NDArray[complex],
        max_iter: int = 10,
    ):
        self.device = get_device(x)
        self.cost = cost  # cost function to be optimized
        self.x = x  # solution to be optimized
        self.dx = dx  # solution update step
        super().__init__(max_iter)

    def _update(self):
        xp = self.device.xp
        ok = self.cost(self.x + self.dx) < self.cost(self.x)
        not_ok = xp.logical_not(ok)
        self.x[ok] += self.dx[ok]
        self.dx[not_ok] /= 10.0

    def _done(self):
        return self.iter >= self.max_iter


class IgnmCauchy(Alg):
    """
    Generic Iterative Gauss-Newton Method (IGNM) algorithm with Cauchy stabilization.

    This class accepts any nonlinear operator (implementing the NonlinearOperator
    interface) along with custom initialization and postprocessing routines.

    The method iteratively solves a linearized problem of the form

        [A'(x)^H A'(x) + α I] dx = A'(x)^H (y - A(x)) + α (x0 - x),

    and then updates x ← x + dx. Here α is a regularization parameter that decays
    over outer iterations.

    Parameters
    ----------
    A : NonLinop
        The nonlinear operator A.
    b : NDArray
        Observation.
    x : ArrayLike
        Variable.
    max_iter : int, optional
        Number of outer (Gauss-Newton) iterations (default is ``10``).
    alpha0 : float, optional
        Initial regularization parameter (default is ``1.0``).
    alpha_min : float, optional
        Minimum regularization parameter (default is ``1e-6``).
    q : float, optional
        Decay factor for α per outer iteration (default is ``2/3``).

    """

    def __init__(
        self,
        A,
        x,
        alpha,
        proxg=None,
        accelerate=False,
        max_iter=100,
        tol=0,
    ):
        self.gradf = gradf
        self.alpha = alpha
        self.accelerate = accelerate
        self.proxg = proxg
        self.x = x
        self.tol = tol

        self.device = backend.get_device(x)
        with self.device:
            if self.accelerate:
                self.z = self.x.copy()
                self.t = 1

        self.resid = np.inf
        super().__init__(max_iter)

    def setup_solver(self):  # noqa
        # Compute right hand side
        rhs = self.A.forward() - self.b

        # Compute jacobian
        JB, JR = self.A.jacobian()

        # Compute gradient G = J^T * r
        bias = self.x - self.x0
        gB = (JB @ rhs).real + self.muB**2 * bias.real
        gR = (JR @ rhs).real + self.muR**2 * bias.imag
        G = gB + 1j * gR  # Complex gradient

        # Compute approximate Hessian
        H1 = (JB @ JB).real + self.muB**2
        H2 = (JB @ JR).real
        H3 = (JR @ JR).real + self.muR**2

        # Compute Cauchy step size
        GG = gB**2 + gR**2
        GHG = gB**2 * H1 + 2 * gB * gR * H2 + gR**2 * H3

        # Stabilize step size
        damp = self._weighted_mean(GHG) / 1000
        step = GG / (GHG + damp) / self.linesearch_iter
        dx = -step * G

        self.cost = LineSearch(costfun, self.x, dx, self.x0)

    def run_solver(self):  # noqa
        raise NotImplementedError

    def _update(self):
        xp = self.device.xp
        self.A.update(self.x)  # x = psi

        # Compute right hand side
        rhs = self.A.forward() - self.b

        # Compute jacobian
        JB, JR = self.A.jacobian()

        # Compute gradient G = J^T * r
        bias = self.x - self.x0
        gB = (JB @ rhs).real + self.muB**2 * bias.real
        gR = (JR @ rhs).real + self.muR**2 * bias.imag
        G = gB + 1j * gR  # Complex gradient

        # Compute approximate Hessian
        H1 = (JB @ JB).real + self.muB**2
        H2 = (JB @ JR).real
        H3 = (JR @ JR).real + self.muR**2

        # Compute Cauchy step size
        GG = gB**2 + gR**2
        GHG = gB**2 * H1 + 2 * gB * gR * H2 + gR**2 * H3

        # Stabilize step size
        damp = self._weighted_mean(GHG) / 1000
        step = GG / (GHG + damp) / self.linesearch_iter
        dx = -step * G

        # Define cost function
        def cost(arg):
            return (
                np.sum(np.abs(pclsr(arg, te, data, opts)) ** 2)
                + opts["muB"] ** 2 * np.sum(np.real(transform(arg, opts) - PSI) ** 2)
                + opts["muR"] ** 2 * np.sum(np.imag(transform(arg, opts) - PSI) ** 2)
            )

        # Basic line search
        for _ in range(self.linesearch_iter):
            ok = cost(self.x + dx) < cost(self.x)
            not_ok = xp.logical_not(ok)
            self.x[ok] += dx[ok]
            self.x[not_ok] += dx[not_ok]

    def _done(self):
        return (self.iter >= self.max_iter) or self.resid <= self.tol
