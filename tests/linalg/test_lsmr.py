"""LSMR solver test."""

import numpy as np

from mrops.linalg import lsmr


def test_lsmr_basic(simple_system):
    """Test LSMR solver on a simple system."""
    A, b, x0 = simple_system
    x = lsmr(A, b, max_iter=10, tol=1e-6)
    np.testing.assert_allclose(x, x0, atol=1e-6)


def test_lsmr_with_damping(simple_system, damp):
    """Test LSMR solver with damping (Tikhonov regularization)."""
    A, b, x0 = simple_system
    x = lsmr(A, b, damp=damp, max_iter=10, tol=1e-6)
    np.testing.assert_allclose(x, x0, atol=1e-2)


def test_lsmr_with_regularizer(simple_system, damp, regularizer):
    """Test LSMR solver with a custom regularization matrix."""
    A, b, x0 = simple_system
    x = lsmr(A, b, damp=damp, R=regularizer, max_iter=10, tol=1e-6)
    np.testing.assert_allclose(x, x0, atol=1e-2)


def test_lsmr_with_bias(simple_system, damp, bias):
    """Test LSMR solver with bias term in regularization."""
    A, b, x0 = simple_system
    x = lsmr(A, b, damp=damp, bias=bias, max_iter=10, tol=1e-6)
    np.testing.assert_allclose(x, x0, atol=1e-2)


def test_lsmr_with_initial_guess(simple_system, x_init):
    """Test LSMR solver with an initial guess."""
    A, b, x0 = simple_system
    x = lsmr(A, b, x=x_init, max_iter=10, tol=1e-6)
    np.testing.assert_allclose(x, x0, atol=1e-6)
