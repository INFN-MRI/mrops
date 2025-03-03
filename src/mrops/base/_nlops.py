"""Nonlinear operator class."""

__all__ = ["NonLinop"]

import abc


class NonLinop(abc.ABC):
    """
    Abstract base class for nonlinear operators in an iterative optimization setting.

    This class follows a structured interface where:
    - `update(x_new)`: Recomputes the forward model F(x) and its Jacobian at x_new.
    - `forward()`: Returns the cached forward operator F(x) as a SigPy Linop.
    - `jacobian()`: Returns the cached Jacobian dF(x) as a SigPy Linop.

    This design ensures compatibility with iterative solvers using SigPy.

    Attributes
    ----------
    x : array-like
        Current operating point for the nonlinear operator.
    F_x : sp.linop.Linop
        Cached forward operator F(x).
    DF_x : sp.linop.Linop
        Cached Jacobian operator dF(x).
    """

    def __init__(self):
        """
        Initialize the nonlinear operator with an initial guess x0.

        Parameters
        ----------
        x0 : array-like
            Initial guess for the variable x.
        """
        self.x = None
        self.F_n = None
        self.DF_n = None

    def update(self, x_new):
        """
        Update the nonlinear operator at a new point x_new.
        This forces recalculation of the forward operator F(x) and its Jacobian dF(x).

        Parameters
        ----------
        x_new : array-like
            New point where the nonlinear operator should be evaluated.
        """
        self.x = x_new
        self.F_n = self._compute_forward(x_new)
        self.DF_n = self._compute_jacobian(x_new)

    def forward(self):
        """
        Returns the current forward operator F(x) as a SigPy Linop.

        Returns
        -------
        sp.linop.Linop
            The forward operator evaluated at the last update point.
        """
        return self.F_n

    def jacobian(self):
        """
        Returns the current Jacobian operator dF(x) as a SigPy Linop.

        Returns
        -------
        sp.linop.Linop
            The Jacobian operator evaluated at the last update point.
        """
        return self.DF_n

    @abc.abstractmethod
    def _compute_forward(self, x):
        """
        Compute the forward operator F(x). Must be implemented by subclasses.

        Parameters
        ----------
        x : array-like
            The input variable at which F(x) should be computed.

        Returns
        -------
        sp.linop.Linop
            SigPy Linop representing the forward model.
        """
        pass

    @abc.abstractmethod
    def _compute_jacobian(self, x):
        """
        Compute the Jacobian operator dF(x). Must be implemented by subclasses.

        Parameters
        ----------
        x : array-like
            The input variable at which dF(x) should be computed.

        Returns
        -------
        sp.linop.Linop
            SigPy Linop representing the Jacobian.
        """
        pass
