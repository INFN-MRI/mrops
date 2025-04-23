"""Common solvers fixtures."""

import pytest

import numpy as np

from mrops import _sigpy as sp


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
    return A, A_matrix @ x, x


@pytest.fixture
def matrix_system():
    """Fixture providing a simple linear system for testing."""
    A = np.array([[4, 1], [1, 3]], dtype=complex)
    x = np.array([1, 2], dtype=complex)
    return A, A @ x, x


@pytest.fixture
def damp():
    return 0.001


@pytest.fixture
def regularizer():
    R_matrix = np.array([[1, 0], [0, 2]], dtype=complex)  # Diagonal regularizer
    return MockLinearOperator(R_matrix)


@pytest.fixture
def matrix_reg():
    R_matrix = np.array([[1, 0], [0, 2]], dtype=complex)  # Diagonal regularizer
    return R_matrix


@pytest.fixture
def bias():
    return np.array([0.5, -0.5], dtype=complex)


@pytest.fixture
def x_init():
    return np.array([0.1, 0.1], dtype=complex)
