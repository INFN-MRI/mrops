"""Iteratively Renormalized Gauss Newton Method with Conjugate Gradient solver."""

__all__ = ["IrgnmCG"]

import gc

from numpy.typing import ArrayLike

from mrinufft._array_compat import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp

from .._sigpy import get_array_module, get_device
from .._sigpy.app import App
from .._sigpy.linop import Identity
from ..base import NonLinop

from ._cg import ConjugateGradient
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
    y : ArrayLike
        Observation.
    x : ArrayLike
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
        b: ArrayLike,
        x: ArrayLike,
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
        b: ArrayLike,
        x: ArrayLike,
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

        # Setup rhs (right hand side)
        b = DGn.H.apply(self.b - Gn.apply(self.x[0])) + self.alpha_n * (
            self.x0 - self.x
        )

        # Setup Operator
        A = DGn.H * DGn + self.alpha_n * Identity(DGn.H.oshape)

        # Initialize CG variables
        device = get_device(self.x)
        if device.id >= 0:
            with device:
                dx0 = device.xp.zeros_like(self.x)
        else:
            dx0 = device.xp.zeros_like(self.x)

        self.solver = ConjugateGradient(
            A, b, dx0, max_iter=self.cg_iter, tol=self.cg_tol
        )

    def run_solver(self):  # noqa
        return self.solver.run()  # return dx
