"""Conjugate Gradient solver test."""

import pytest

import numpy as np

from mrops import _sigpy as sp
from mrops.linalg import cg


class MockLinearOperator(sp.linop.Linop):
    def __init__(self, matrix):
        self._matrix = matrix
        super().__init__([self._matrix.shape[0]], [self._matrix.shape[1]])

    def _apply(self, input):
        return self._matrix @ input

    def _adjoint_linop(self):
        return self.__class__(self._matrix.conj().T)

    def _normal_linop(self):
        return self.__class__(self._matrix.conj().T @ self._matrix)


@pytest.fixture
def simple_system():
    """Fixture providing a simple linear system for testing."""
    A_matrix = np.array([[4, 1], [1, 3]], dtype=complex)
    A = MockLinearOperator(A_matrix)  # Convert to sigpy Linop
    x = np.array([1, 2], dtype=complex)
    return A, x, A_matrix @ x


def test_cg_basic(simple_system):
    """Test CG solver on a simple system."""
    A, x0, b = simple_system
    x = cg(A, b, max_iter=10, tol=1e-6)
    np.testing.assert_allclose(x, x0, atol=1e-6)


def test_cg_with_damping(simple_system):
    """Test CG solver with damping (Tikhonov regularization)."""
    A, x0, b = simple_system
    damp = 0.001
    x = cg(A, b, damp=damp, max_iter=10, tol=1e-6)

    # With damping, the solution won't be exactly x0, but close.
    np.testing.assert_allclose(x, x0, atol=1e-2)


def test_cg_with_regularizer(simple_system):
    """Test CG solver with a custom regularization matrix."""
    A, x0, b = simple_system
    R_matrix = np.array([[1, 0], [0, 2]], dtype=complex)  # Diagonal regularizer
    R = MockLinearOperator(R_matrix)
    damp = 0.001
    x = cg(A, b, damp=damp, R=R, max_iter=10, tol=1e-6)

    # Regularization changes the solution, should be close but different from x0
    np.testing.assert_allclose(x, x0, atol=1e-2)


def test_cg_with_bias(simple_system):
    """Test CG solver with bias term in regularization."""
    A, x0, b = simple_system
    bias = np.array([0.5, -0.5], dtype=complex)
    damp = 0.001
    x = cg(A, b, damp=damp, bias=bias, max_iter=10, tol=1e-6)

    # Solution should be between x0 and bias, as bias pulls x towards it
    np.testing.assert_allclose(x, x0, atol=1e-2)


def test_cg_with_initial_guess(simple_system):
    """Test CG solver with an initial guess."""
    A, x0, b = simple_system
    x_init = np.array([0.1, 0.1], dtype=complex)
    x = cg(A, b, x=x_init, max_iter=10, tol=1e-6)

    np.testing.assert_allclose(x, x0, atol=1e-6)
