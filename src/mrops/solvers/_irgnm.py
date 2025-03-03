"""Iteratively Renormalized Gauss Newton Method."""

__all__ = ["IrgnmBase"]

from numpy.typing import ArrayLike

from .._sigpy.alg import Alg
from ..base import NonLinop


class IrgnmBase(Alg):
    """
    Generic Iteratively Regularized Gauss-Newton Method (IRGNM) algorithm.

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
    b : ArrayLike
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
        A: NonLinop,
        b: ArrayLike,
        x: ArrayLike,
        max_iter: int = 10,
        alpha0: float = 1.0,
        alpha_min: float = 1e-6,
        q: float = 2 / 3,
    ):
        self.A = A
        self.b = b
        self.x = x
        self.alpha0 = alpha0
        self.alpha_min = alpha_min
        self.q = q
        self.alpha_n = None
        self.x0 = x.copy()

        super().__init__(max_iter)

    def setup_solver(self):  # noqa
        raise NotImplementedError

    def run_solver(self):  # noqa
        raise NotImplementedError

    def _update(self):
        """
        Run the IRGNM algorithm to solve A(x) = b.

        Parameters
        ----------
        b : np.ndarray
            Measured data in the data domain.

        Returns
        -------
        x : np.ndarray
            Final estimate after (optional) postprocessing.

        """
        self.alpha_n = max(self.alpha0 * (self.q**self.iter), self.alpha_min)
        self.A.update(self.x)
        self.setup_solver()

        # Perform inner loop
        dx = self.run_solver()

        # Update solution
        self.x = self.x + dx
