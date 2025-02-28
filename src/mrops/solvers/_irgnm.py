"""Iteratively Renormalized Gauss Newton Method."""

__all__ = ["IRGNMBase"]


class IRGNMBase:
    """
    Generic Iteratively Regularized Gauss-Newton Method (IRGNM) algorithm.

    This class accepts any nonlinear operator (implementing the NonlinearOperator
    interface) along with custom initialization and postprocessing routines.

    The method iteratively solves a linearized problem of the form

        [F'(x)^H F'(x) + α I] dx = F'(x)^H (y - F(x)) + α (x0 - x),

    and then updates x ← x + dx. Here α is a regularization parameter that decays
    over outer iterations.

    Parameters
    ----------
    operator : NonlinearOperator
        The nonlinear operator F.
    solver : App
        The inner solver
    maxiter : int, optional
        Number of outer (Gauss-Newton) iterations (default is 10).
    alpha0 : float, optional
        Initial regularization parameter (default is 1.0).
    q : float, optional
        Decay factor for α per outer iteration (default is 2/3).

    """

    def __init__(self, operator, solver, maxiter=10, alpha0=1.0, q=2 / 3):
        self.operator = operator
        self.solver = solver
        self.maxiter = maxiter
        self.alpha0 = alpha0
        self.q = q

    def run(self, y):
        """
        Run the IRGNM algorithm to solve F(x) = y.

        Parameters
        ----------
        y : np.ndarray
            Measured data in the data domain.

        Returns
        -------
        x : np.ndarray
            Final estimate after (optional) postprocessing.
        """
        x0 = self.init_func()
        x = x0.copy()

        for n in range(self.num_outer):
            alpha_n = self.alpha0 * (self.q**n)
            F_x = self.operator.evaluate(x)
            r = y - F_x

            def L(dx):
                return (
                    self.operator.derivative_adjoint(x, self.operator.derivative(x, dx))
                    + alpha_n * dx
                )

            b = self.operator.derivative_adjoint(x, r) + alpha_n * (x0 - x)
            dx = cg_solve(L, b, np.zeros_like(x), maxiter=self.cg_maxiter)
            x = x + dx

            res_norm = np.linalg.norm(b - L(dx))

        if self.postprocess_func is not None:
            return self.postprocess_func(x)
        else:
            return x
