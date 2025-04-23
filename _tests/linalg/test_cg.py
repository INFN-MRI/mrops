"""Conjugate Gradient solver test."""

import numpy as np

from mrops.linalg import cg


def test_cg_basic(simple_system):
    """Test CG solver on a simple system."""
    A, b, x0 = simple_system
    x = cg(A, b, max_iter=10, tol=1e-6)
    np.testing.assert_allclose(x, x0, atol=1e-6)


def test_cg_with_damping(simple_system, damp):
    """Test CG solver with damping (Tikhonov regularization)."""
    A, b, x0 = simple_system
    x = cg(A, b, damp=damp, max_iter=10, tol=1e-6)
    np.testing.assert_allclose(x, x0, atol=1e-2)


def test_cg_with_regularizer(simple_system, damp, regularizer):
    """Test CG solver with a custom regularization matrix."""
    A, b, x0 = simple_system
    x = cg(A, b, damp=damp, R=regularizer, max_iter=10, tol=1e-6)
    np.testing.assert_allclose(x, x0, atol=1e-2)


def test_cg_with_bias(simple_system, damp, bias):
    """Test CG solver with bias term in regularization."""
    A, b, x0 = simple_system
    x = cg(A, b, damp=damp, bias=bias, max_iter=10, tol=1e-6)
    np.testing.assert_allclose(x, x0, atol=1e-2)


def test_cg_with_initial_guess(simple_system, x_init):
    """Test CG solver with an initial guess."""
    A, b, x0 = simple_system
    x = cg(A, b, x=x_init, max_iter=10, tol=1e-6)
    np.testing.assert_allclose(x, x0, atol=1e-6)
