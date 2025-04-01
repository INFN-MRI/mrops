"""Porting of original MATLAB implementation of NLINV"""

__all__ = ["nlinv_matlab", "simu"]

import numpy as np


def nlinv_matlab(Y, n):
    """
    Nonlinear inversion for parallel MRI reconstruction.

    Parameters
    ----------
    Y : ndarray (c, y, x)
        k-space measurements.
    n : int
        Number of nonlinear iterations.

    Returns
    -------
    R : ndarray (n, y, x)
        Reconstructed images.
    """
    print("Start...")

    alpha = 1.0
    c, y, x = Y.shape
    R = np.zeros((n, y, x), dtype=np.complex64)

    # Initialization of x-vector
    X0 = np.zeros((c + 1, y, x), dtype=np.complex64)
    X0[0] = 1.0  # Object part

    # Initialize mask and weights
    P = pattern(Y)
    W = weights(x, y)

    # Normalize data vector
    yscale = 100.0 / np.sqrt(np.vdot(Y, Y))
    YS = Y * yscale

    XN = np.zeros_like(X0)
    XT = np.zeros_like(X0)
    XN[:] = X0

    for i in range(n):
        # Apply weights to XN (excluding first component)
        XT[0] = XN[0]
        for s in range(1, c + 1):
            XT[s] = apweights(W, XN[s])

        RES = YS - op(P, XT)

        print("Residuum:", np.sqrt(np.vdot(RES, RES)))

        # Calculate RHS
        r = derH(P, W, XT, RES)
        r += alpha * (X0 - XN)

        # Conjugate Gradient (CG) initialization
        z = np.zeros_like(r)
        d = np.zeros_like(r)

        dnew = np.vdot(r, r)
        dnot = dnew
        d[:] = r

        for j in range(500):
            # Regularized normal equations
            q = derH(P, W, XT, der(P, W, XT, d)) + alpha * d

            a = dnew / np.real(np.vdot(d, q))
            z += a * d
            r -= a * q

            dold = dnew
            dnew = np.vdot(r, r)

            d = (dnew / dold) * d + r

            print("CG residuum:", dnew / dnot)

            if np.sqrt(dnew) < 1.0e-2 * dnot:
                break

        # End CG
        XN += z
        print(np.sqrt(np.vdot(z, z)))
        print(np.sqrt(np.vdot(XN, XN)))

        alpha /= 3.0

        # Post-processing
        C = np.zeros((y, x), dtype=np.complex64)
        for s in range(1, c + 1):
            CR = apweights(W, XN[s])
            C += np.conj(CR) * CR

        R[i] = XN[0] * np.sqrt(C) / yscale

    return R


def simu(x, y):
    """
    Simulate object and coil sensitivities, then apply undersampling.

    Parameters
    ----------
    x, y : int
        Dimensions of the image.

    Returns
    -------
    R : ndarray (4, y, x)
        Simulated undersampled k-space data.
    """
    X = np.zeros((5, y, x))  # Storage order reversed from MATLAB

    i, j = np.meshgrid(np.arange(y), np.arange(x), indexing="ij")
    d = ((i / y) - 0.5) ** 2 + ((j / x) - 0.5) ** 2

    # Object
    X[0, d < 0.4**2] = 1.0

    # Coil sensitivities
    d1 = ((i / y) - 1.0) ** 2 + ((j / x) - 0.0) ** 2
    d2 = ((i / y) - 1.0) ** 2 + ((j / x) - 1.0) ** 2
    d3 = ((i / y) - 0.0) ** 2 + ((j / x) - 0.0) ** 2
    d4 = ((i / y) - 0.0) ** 2 + ((j / x) - 1.0) ** 2

    X[1] = np.exp(-d1)
    X[2] = np.exp(-d2)
    X[3] = np.exp(-d3)
    X[4] = np.exp(-d4)

    # Undersampling pattern
    P = np.zeros((y, x))
    P[:, ::2] = 1.0  # Every other column
    P[:, (y // 2 - 8) : (y // 2 + 8)] = 1.0  # Center region

    # Simulate k-space data
    return op(P, X), P


# %% utils
def myfft(x):
    """Apply FFT with correct shifting."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))


def myifft(x):
    """Apply IFFT with correct shifting."""
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x), norm="ortho"))


def pattern(Y):
    """
    Compute the sampling pattern mask.

    Parameters
    ----------
    Y : ndarray (..., c, y, x)
        Input k-space data.

    Returns
    -------
    P : ndarray (..., y, x)
        Sampling pattern.
    """
    return np.sum(np.abs(Y) ** 2, axis=-3) > 0


def weights(y, x):
    """
    Compute the k-space weighting operator.

    Parameters
    ----------
    y, x : int
        Dimensions of the k-space.

    Returns
    -------
    W : ndarray (y, x)
        Weighting matrix.
    """
    i, j = np.meshgrid(np.arange(y), np.arange(x), indexing="ij")
    d = ((i / y) - 0.5) ** 2 + ((j / x) - 0.5) ** 2
    return 1.0 / (1.0 + 220.0 * d) ** 16


def apweights(W, CT):
    """
    Apply k-space weighting.

    Parameters
    ----------
    W : ndarray (..., y, x)
        Weighting matrix.
    CT : ndarray (..., y, x)
        Input data in k-space.

    Returns
    -------
    C : ndarray (..., y, x)
        Weighted output.
    """
    return myifft(W * CT)


def apweightsH(W, CT):
    """
    Apply adjoint k-space weighting.

    Parameters
    ----------
    W : ndarray (..., y, x)
        Weighting matrix.
    CT : ndarray (..., y, x)
        Input data in image space.

    Returns
    -------
    C : ndarray (..., y, x)
        Adjoint-weighted output.
    """
    return np.conj(W) * myfft(CT)


def op(P, X):
    """
    Apply forward model operator.

    Parameters
    ----------
    P : ndarray (..., y, x)
        Sampling pattern.
    X : ndarray (..., c+1, y, x)
        Input image data.

    Returns
    -------
    K : ndarray (..., c, y, x)
        Output k-space data.
    """
    K = np.zeros_like(X[..., 1:, :, :], dtype=np.complex64)
    for i in range(X.shape[-3] - 1):
        K[..., i, :, :] = P * myfft(X[..., 0, :, :] * X[..., i + 1, :, :])
    return K


def der(P, W, X0, DX):
    """
    Compute derivative of forward operator.

    Parameters
    ----------
    P : ndarray (..., y, x)
        Sampling pattern.
    W : ndarray (..., y, x)
        Weighting matrix.
    X0 : ndarray (..., c+1, y, x)
        Current estimate.
    DX : ndarray (..., c+1, y, x)
        Perturbation.

    Returns
    -------
    K : ndarray (..., c, y, x)
        Output k-space data.
    """
    K = np.zeros_like(DX[..., 1:, :, :])
    for i in range(DX.shape[-3] - 1):
        K[..., i, :, :] = X0[..., 0, :, :] * apweights(W, DX[..., i + 1, :, :])
        K[..., i, :, :] += DX[..., 0, :, :] * X0[..., i + 1, :, :]
        K[..., i, :, :] = P * myfft(K[..., i, :, :])
    return K


def derH(P, W, X0, DK):
    """
    Compute adjoint derivative.

    Parameters
    ----------
    P : ndarray (..., y, x)
        Sampling pattern.
    W : ndarray (..., y, x)
        Weighting matrix.
    X0 : ndarray (..., c+1, y, x)
        Current estimate.
    DK : ndarray (..., c, y, x)
        Input data in k-space.

    Returns
    -------
    DX : ndarray (..., c+1, y, x)
        Output data.
    """
    DX = np.zeros_like(X0)
    for i in range(DK.shape[-3]):
        K = myifft(P * DK[..., i, :, :])
        DX[..., 0, :, :] += K * np.conj(X0[..., i + 1, :, :])
        DX[..., i + 1, :, :] = apweightsH(W, K * np.conj(X0[..., 0, :, :]))
    return DX
