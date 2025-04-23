"""LSTSQ solver test."""

import numpy as np

from mrops.linalg import lstsq


def test_lstsq_basic(matrix_system):
    """Test LSTSQ solver on a simple system."""
    A, b, x0 = matrix_system
    x = lstsq(A, b)
    np.testing.assert_allclose(x, x0, atol=1e-6)


def test_lstsq_with_damping(matrix_system, damp):
    """Test LSTSQ solver with damping (Tikhonov regularization)."""
    A, b, x0 = matrix_system
    x = lstsq(A, b, damp=damp)
    np.testing.assert_allclose(x, x0, atol=1e-2)


def test_lstsq_with_regularizer(matrix_system, damp, matrix_reg):
    """Test LSTSQ solver with a custom regularization matrix."""
    A, b, x0 = matrix_system
    x = lstsq(A, b, damp=damp, R=matrix_reg)
    np.testing.assert_allclose(x, x0, atol=1e-2)


def test_lstsq_with_bias(matrix_system, damp, bias):
    """Test LSTSQ solver with bias term in regularization."""
    A, b, x0 = matrix_system
    x = lstsq(A, b, damp=damp, bias=bias)
    np.testing.assert_allclose(x, x0, atol=1e-2)


# def test_lstsq_with_constraints():
#     """Test equality-constrained least squares solver."""
#     # Define a small least squares problem
#     A = np.asarray([[3.0, 2.0], [1.0, 4.0], [2.0, 1.0]])
#     b = np.asarray([1.0, 2.0, 3.0])

#     # Define constraints Bx = d
#     C = np.asarray([[1.0, 1.0]])  # Enforces x1 + x2 = 1
#     d = np.asarray([1.0])

#     # Solve the constrained least squares problem
#     x = lstsq(A, b, C=C, d=d)

#     # Verify constraints hold: Bx ≈ d
#     np.testing.assert_allclose(C @ x, d, atol=1e-6)

#     # Verify least squares approximation: Ax ≈ b
#     b_pred = A @ x
#     np.testing.assert_allclose(b_pred, b, atol=1e-2)  # Slight tolerance for LS error
