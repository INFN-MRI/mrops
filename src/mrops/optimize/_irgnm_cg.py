"""Iteratively Renormalized Gauss Newton Method with Conjugate Gradient solver."""

__all__ = ["IrgnmCG"]

import gc

from numpy.typing import NDArray

from mrinufft._array_compat import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp

from .._sigpy.app import App

from ..base import NonLinop
from ..linalg import ConjugateGradient
from ._irgnm import IrgnmBase


class IrgnmCG(App):
    """
    Conjugate Gradient Iteratively Regularized Gauss-Newton Method (IRGNM) algorithm.

    This class accepts any nonlinear operator (implementing the NonlinearOperator
    interface) along with custom initialization and postprocessing routines.

    The method iteratively solves a linearized problem of the form

        [F'(x)^H F'(x) + α I] dx = F'(x)^H (y - F(x)) + α (x0 - x),

    using Conjugate Gradient algorithm, and then updates x ← x + dx.
    Here α is a regularization parameter that decays over outer iterations.

    Parameters
    ----------
    A : NonLinop
        The nonlinear operator A.
    y : NDArray[complex | float]
        Observation.
    x : NDArray[comples | float]
        Variable.
    max_iter : int, optional
        Number of outer (Gauss-Newton) iterations (default is ``10``).
    cg_iter : int, optional
        Number of inner (Conjugate Gradient) iterations (default is ``10``).
    cg_tol : float, optional
         Tolerance for Conjugate Gradient stopping condition (default is ``0.0``).
    alpha0 : float, optional
        Initial regularization parameter (default is ``1.0``).
    alpha_min : float, optional
        Minimum regularization parameter (default is ``1e-6``).
    q : float, optional
        Decay factor for α per outer iteration (default is ``2/3``).
    show_pbar : bool, optional
        Toggle whether show progress bar (default is ``False``).
    leave_pbar : bool, optional
        Toggle whether to leave progress bar after finished (default is ``True``).
    record_time : bool, optional
        Toggle wheter record runtime (default is ``False``).

    """

    def __init__(
        self,
        A: NonLinop,
        b: NDArray[complex | float],
        x: NDArray[complex | float],
        max_iter: int = 10,
        cg_iter: int = 20,
        cg_tol: float = 1e-2,
        alpha0: float = 1.0,
        alpha_min: float = 0.0,
        q: float = 2 / 3,
        show_pbar: bool = False,
        leave_pbar: bool = True,
        record_time: bool = False,
    ):
        _alg = _IrgnmCG(A, b, x, max_iter, cg_iter, cg_tol, alpha0, alpha_min, q)
        super().__init__(_alg, show_pbar, leave_pbar, record_time)

    def _output(self):
        gc.collect()
        if CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()
        return self.alg.x


class _IrgnmCG(IrgnmBase):
    def __init__(
        self,
        A: NonLinop,
        b: NDArray[complex | float],
        x: NDArray[complex | float],
        max_iter: int = 10,
        cg_iter: int = 20,
        cg_tol: float = 1e-2,
        alpha0: float = 1.0,
        alpha_min: float = 0.0,
        q: float = 2 / 3,
    ):
        super().__init__(A, b, x, max_iter, alpha0, alpha_min, q)
        self.cg_iter = cg_iter
        self.cg_tol = cg_tol

    def setup_solver(self):  # noqa
        Gn = self.A.forward()
        DGn = self.A.jacobian()

        # initialize solver
        self.solver = ConjugateGradient(
            DGn,
            self.b - Gn.apply(self.x[0]),
            damp=self.alpha_n,
            bias=self.x0 - self.x,
            max_iter=self.cg_iter,
            tol=self.cg_tol,
        )

    def run_solver(self):  # noqa
        return self.solver.run()  # return dx
