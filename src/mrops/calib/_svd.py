"""Utilities for SVD-based subspace estimation (e.g., Bloch or Coil Compression)."""

from __future__ import annotations

__all__ = [
    "SVDCompression",
    "compress_coil",
    "estimate_coil_subspace",
    "estimate_bloch_subspace",
]

import torch

from numpy.typing import NDArray

from mrinufft._array_compat import with_torch

from ._acr import extract_acr


def estimate_bloch_subspace(
    training_data: NDArray[complex],
    num_coeff: int = None,
    variance_ratio: float = None,
) -> SVDCompression:
    """
    Estimate Bloch subspace basis.

    Parameters
    ----------
    training_data : NDArray[complex]
        Training dataset of shape ``(num_atoms, num_contrasts)``.
    num_coeff : int, optional
        Size of subspace basis (i.g., subspace size). User can either specify this
        or the target explained variance ratio.
    explained_variance_ratio : float, optional
        Explained variance ratio for the given basis size. User can either specify this
        or the desired subspace size.

    Returns
    -------
    basis : SVDCompression
        SVD compression operator.

    """
    return SVDCompression(training_data, num_coeff, variance_ratio).basis


def estimate_coil_subspace(
    data: NDArray[complex],
    num_coils: int = None,
    variance_ratio: float = None,
    cal_width: int = 24,
    ndim: int = None,
    coords: NDArray[complex] = None,
    shape: int = None,
    coil_axis: int = None,
) -> SVDCompression:
    """
    Estimate Bloch subspace basis.

    Parameters
    ----------
    training_data : NDArray[complex]
        Training dataset of shape ``(..., num_atoms, space_size)``.
    num_coeff : int, optional
        Size of subspace basis (i.g., subspace size).
        User can either specify this or the target explained variance ratio.
    explained_variance_ratio : float, optional
        Explained variance ratio for the given basis size. User can either specify this
        or the desired subspace size.
    coil_axis : int, optional
        Coil dimension axis. If none, assume is the first on the left of spatial
        encoding dimensions.

    Returns
    -------
    basis : SVDCompression
        SVD compression operator.

    """
    if coil_axis is None:
        axis = -coords.ndim if coords is not None else 0
    else:
        axis = coil_axis

    # if axis is not 0, this is batched
    if len(data.shape[:axis]) != 0:
        batched = True
    else:
        batched = False

    # extract calibration region
    if coords is None:
        training_data = extract_acr(
            data,
            cal_width,
            ndim=ndim,
        )
        # reshape training data to (num_samples, num_coils)
        training_data = training_data[..., None].swapaxes(axis - 1, -1)
    else:
        ndim = coords.shape[-1]
        training_data, _, _ = extract_acr(data, cal_width, coords=coords, shape=shape)
        training_data = training_data[..., None].swapaxes(axis - 1, -1)

    if batched:
        training_data = training_data.reshape(
            training_data.shape[0], -1, training_data.shape[-1]
        )
    else:
        training_data = training_data.reshape(-1, training_data.shape[-1])

    return SVDCompression(training_data, num_coils, variance_ratio)


def compress_coil(
    data: NDArray[complex],
    num_coils: int = None,
    variance_ratio: float = None,
    cal_width: int = 24,
    ndim: int = None,
    coords: NDArray[float] = None,
    shape: int = None,
) -> NDArray[complex]:
    """
    Compress data along ``coil`` axis.

    Parameters
    ----------
    data : NDArray[complex]
        Input k-space dataset of shape ``(B, C, ...)`` (2D Cartesian),
        ``(C, ...)`` (3D Cartesian), ``(B, C, *trajectory.shape[:-1])`` (2D Non Cartesian)
        or ``(C, *trajectory.shape[:-1])`` (3D Non Cartesian).
    num_coils : int, optional
        Number of virtual channels. User can either specify this
        or the target explained variance ratio.
    explained_variance_ratio : float, optional
        Explained variance ratio for the given number of virtual channels.
        User can either specify this or the desired subspace size.
    cal_width : int, optional
        Calibration region size. The default is ``24``.
    ndim : int, optional
        Number of spatial dimensions. The default is ``None``.
        Required for Cartesian datasets.
    coords : NDArray[float], optional
        K-space trajectory of shape ``(num_shots, num_pts, ndim)``, normalized between ``(-0.5, 0.5)``.
        Required for Non Cartesian datasets. The default is ``None``.
    shape : int, optional
        Matrix size of shape ``(ndim,)``.
        Required for Non Cartesian datasets. The default is ``None``.

    Returns
    -------
    NDArray[complex]
        Compressed k-space.

    """
    axis = -coords.ndim if coords is not None else 0
    basis = estimate_coil_subspace(
        data, num_coils, variance_ratio, cal_width, ndim, coords, shape
    )

    return basis(data, axis=axis)


class SVDCompression:
    """
    Subspace basis estimator via SVD.

    Parameters
    ----------
    training_data : ArrayLike
        Training data for subspace estimation of shape ``(num_samples, space_size)``.
    num_coeff : int, optional
        Size of subspace basis (i.g., subspace size). User can either specify this
        or the target explained variance ratio.
    explained_variance_ratio : float, optional
        Explained variance ratio for the given basis size. User can either specify this
        or the desired subspace size.

    Attributes
    ----------
    basis : ArrayLike
        Subspace basis of shape ``(space_size, subspace_size)``
    explained_variance_ratio : float
        Explained variance ratio for the given basis size.

    """

    @with_torch
    def __init__(
        self,
        training_data: NDArray[complex],
        num_coeff: int = None,
        variance_ratio: float = None,
    ):
        if num_coeff is None and variance_ratio is None:
            raise ValueError("Please specify 'num_coeff' or 'variance_ratio'.")
        if num_coeff is not None and variance_ratio is not None:
            raise ValueError(
                "Please either specify 'num_coeff' or 'variance_ratio', not both."
            )

        # Perform SVD compression
        U, S, Vh = torch.linalg.svd(training_data, full_matrices=False)

        # Get variance
        num_samples = training_data.shape[-2]
        explained_variance = S**2 / (num_samples - 1)
        total_variance = explained_variance.sum(axis=-1)
        if explained_variance.ndim > 1:
            while total_variance.ndim < explained_variance.ndim:
                total_variance = total_variance[..., None]
        self._explained_variance_ratio = explained_variance / total_variance

        # Get output coefficients from variance ratio.
        if variance_ratio is not None:
            cum_variance = torch.cumsum(self._explained_variance_ratio, -1)
            self._num_coeff = (cum_variance <= variance_ratio).sum(axis=-1).max().item()
            self._num_coeff = max(1, self._num_coeff)
        else:
            self._num_coeff = num_coeff
            if self._num_coeff > training_data.shape[-1]:
                raise ValueError(
                    f"Requested number of coefficients {num_coeff} larger than space {training_data.shape[-1]}"
                )

        self._basis = Vh.swapaxes(-1, -2).conj()[..., : self._num_coeff]

    @with_torch
    def __call__(self, input, axis=0):  # noqa
        space_size = input.shape[axis]

        # apply compression
        if len(self._basis.shape) == 2:
            _output = input.swapaxes(axis, -1)
            _tmp_shape = _output.shape
            _output = _output.reshape(-1, space_size)
            _output = _output @ self._basis
        else:
            _output = input.swapaxes(axis, -1)
            _tmp_shape = _output.shape
            _output = _output.reshape(_tmp_shape[0], -1, space_size)
            _output = torch.einsum("nki,nij->nkj", _output, self._basis)

        # reshape
        _output = _output.reshape(*_tmp_shape[:-1], self._num_coeff).swapaxes(axis, -1)
        return _output.contiguous()

    @property
    def basis(self):  # noqa
        return self._basis

    @property
    def num_coeff(self):  # noqa
        return self._num_coeff

    @property
    def explained_variance_ratio(self):  # noqa
        return self._explained_variance_ratio
