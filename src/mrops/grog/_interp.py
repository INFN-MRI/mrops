"""GRAPPA operator based interpolation. Adapted for convenience from PyGRAPPA."""

__all__ = ["interp"]

import warnings

from numpy.typing import NDArray

import numpy as np
import numba as nb

from scipy.linalg import fractional_matrix_power as fmp
from scipy.spatial import KDTree

from mrinufft._array_compat import with_numpy

from .._sigpy import get_device, to_device
from .._utils import rescale_coords


def interp(
    interpolator: dict,
    input: NDArray[complex],
    coords: NDArray[float],
    shape: list[int] | tuple[int],
    stack_axes: list[int] | tuple[int] | None = None,
    oversamp: float | list[float] | tuple[float] | None = None,
    radius: float = 0.75,
    precision: int = 1,
    weighting_mode: str = "distance",
) -> tuple[NDArray[complex], NDArray[int], tuple[int]]:
    """
    GRAPPA Operator Gridding (GROP) interpolation of Non-Cartesian datasets.

    Parameters
    ----------
    input : NDArray[complex]
        Input Non-Cartesian kspace.
    coords : NDArray[float]
        Fourier domain coordinates array of shape ``(..., ndims)``.
    shape : list[int] | tuple[int]
        Cartesian grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    interpolator: dict
        Trained GROG interpolator.
    stack_axes: list[int] | tuple[int], optional
        Index marking stack axes. The default is ``None``,
    oversamp: float | list[float] | tuple[float] | None, optional
        Cartesian grid oversampling factor. If scalar, assume
        same oversampling for all spatial dimensions.
        The default is ``1.0`` (2D MRI) or ``(1.0, 1.0, 1.2)`` (3D MRI).
    radius: float, optional
        Spreading radius. The default is ``0.75``.
    precision: int, optional
        Number of decimal digits in GROG kernel power. The default is ``1``.
    weighting_mode: str, optional
        Non Cartesian samples accumulation mode. Can be:

            * ``"average"``: arithmetic average.
            * ``"distance"``: weight according to distance.

        The default is ``"distance"``.

    Returns
    -------
    output : NDArray[complex]
        Output sparse Cartesian kspace of shape ``(N,)``.
    indexes : NDArray[int]
        Sampled k-space points indexes of shape ``(N, 2)``,
        with rightmost axis being ``(stack_idx, grid_idx)``.
    shape : tuple[int]
        Oversampled k-space grid size.

    Notes
    -----
    Produces the unit operator described in [1]_.

    This seems to only work well when coil sensitivities are very
    well separated/distinct.  If coil sensitivities are similar,
    operators perform poorly.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Parallel magnetic resonance
           imaging using the GRAPPA operator formalism." Magnetic
           resonance in medicine 54.6 (2005): 1553-1556.

    """
    device = get_device(input)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        data, indexes, shape = _interp(
            interpolator,
            input,
            coords,
            shape,
            stack_axes,
            oversamp,
            radius,
            precision,
            weighting_mode,
        )

    # enforce correct device
    data = to_device(data, device)
    indexes = to_device(indexes, device)

    return data, indexes, shape


# %% subroutines
@with_numpy
def _interp(
    interpolator,
    input,
    coords,
    shape,
    stack_axes,
    oversamp,
    radius,
    precision,
    weighting_mode,
):
    if radius > 1.0:
        raise ValueError(f"Maximum GRAPPA shift is 1.0, requested {radius}")
    if weighting_mode.lower() in ["average", "distance"] is False:
        raise ValueError(
            f"Weighting mode can be either 'average' or 'distance', requested {weighting_mode}"
        )
    weighting_mode = weighting_mode.lower()
    stack_axes = check_stack(stack_axes)

    # calculate interpolation stepsize
    pfac = 10.0**precision
    radius = np.ceil(pfac * radius) / pfac
    radius = radius.item()
    nsteps = 2 * radius / 10 ** (-precision) + 1
    nsteps = int(nsteps)

    # compute exponends
    deltas = 2 * radius * (np.linspace(0, 1, nsteps) - 0.5)

    # pre-compute partial operators
    Dx = _grog_power(interpolator["x"], deltas).astype(input.dtype)  # (nsteps, nc, nc)
    Dy = _grog_power(interpolator["y"], deltas).astype(input.dtype)  # (nsteps, nc, nc)
    if "z" in interpolator and interpolator["z"] is not None:
        Dz = _grog_power(interpolator["z"], deltas).astype(
            input.dtype
        )  # (nsteps, nc, nc), 3D only
    else:
        Dz = None

    if stack_axes is None:
        stack_shape = ()
    else:
        stack_shape = coords.shape[: len(stack_axes)]
    signal_shape = coords.shape[len(stack_shape) : -1]
    ndim = coords.shape[-1]

    # expand oversamp
    if oversamp is None:
        if ndim == 2:
            oversamp = 1.0
        else:
            oversamp = [1.0, 1.0, 1.2]
    if np.isscalar(oversamp):
        oversamp = tuple(ndim * [oversamp])
    elif len(oversamp) == 1:
        oversamp = tuple(ndim * oversamp[0])
    else:
        oversamp = tuple(oversamp)

    # get batch shape
    batch_shape = input.shape[: -len(signal_shape) - len(stack_shape)]

    # get number of batches, number of stack and number of coils (assume it is the rightmost batch)
    n_batches = int(np.prod(batch_shape[:-1]))
    n_stacks = int(np.prod(stack_shape))
    n_samples = int(np.prod(signal_shape))
    n_coils = batch_shape[-1]

    # reshape data to (nsamples, nbatches, ncoils)
    input = input.reshape(n_batches, n_coils, -1)
    input = input.transpose(2, 0, 1)
    input = np.ascontiguousarray(input)

    # generate stack coordinate
    if len(stack_shape) > 0:
        stack_coords = np.meshgrid(
            *[np.arange(0, stack_size) for stack_size in stack_shape],
            indexing="ij",
        )
        stack_coords = np.stack([ax.ravel() for ax in stack_coords], axis=-1)
        stack_coords = np.repeat(stack_coords, n_samples, axis=0)
        stack_coords = stack_coords.astype(int)
        stack_coords = stack_coords * np.asarray(stack_shape[1:] + (1,), dtype=int)
        stack_coords_flat = stack_coords.sum(axis=-1)
    else:
        stack_coords = np.zeros(n_samples, dtype=int)
        stack_coords_flat = stack_coords

    # reshape coordinates
    coords = coords.reshape(-1, ndim)
    coords = rescale_coords(coords, shape)

    # build target
    grid = np.meshgrid(
        *[
            np.linspace(
                -shape[n] // 2, shape[n] // 2 - 1, int(np.ceil(oversamp[n] * shape[n]))
            )
            for n in range(ndim)
        ],
        indexing="ij",
    )
    grid = np.stack([ax.ravel() for ax in grid], axis=-1).astype(np.float32)

    # remove indices outside support
    # inside = (grid**2).sum(axis=-1)**0.5 <= 0.5 * np.max(shape)
    # grid = np.ascontiguousarray(grid[inside, :])

    # find source for each target
    noncart_indices, weights, grog_index, indexes, bin_starts, bin_counts = kdtree(
        n_stacks,
        grid,
        coords,
        stack_coords_flat,
        radius,
        precision,
        weighting_mode,
    )

    # raise grog operators to required power
    Dx = np.asarray(Dx)
    Dy = np.asarray(Dy)
    if Dz is not None:
        Dz = np.asarray(Dz)

    # precompute grog table
    if ndim == 2:
        Dx = Dx[None, :, ...]
        Dy = Dy[:, None, ...]
        Dx = np.repeat(Dx, nsteps, axis=0)  # (nsteps, nsteps, nc, nc)
        Dy = np.repeat(Dy, nsteps, axis=1)  # (nsteps, nsteps, nc, nc)
        Dx = Dx.reshape(-1, *Dx.shape[-2:])  # (nsteps**2, nc, nc)
        Dy = Dy.reshape(-1, *Dy.shape[-2:])  # (nsteps**2, nc, nc)
        grog_table = Dx @ Dy  # (nsteps**2, nc, nc)
    elif ndim == 3:
        Dx = Dx[None, None, :, ...]
        Dy = Dy[None, :, None, ...]
        Dz = Dz[:, None, None, ...]
        Dx = np.repeat(Dx, nsteps, axis=0)  # (nsteps, nsteps, nsteps, nc, nc)
        Dx = np.repeat(Dx, nsteps, axis=1)  # (nsteps, nsteps, nsteps, nc, nc)
        Dy = np.repeat(Dy, nsteps, axis=0)  # (nsteps, nsteps, nsteps, nc, nc)
        Dy = np.repeat(Dy, nsteps, axis=2)  # (nsteps, nsteps, nsteps, nc, nc)
        Dz = np.repeat(Dz, nsteps, axis=1)  # (nsteps, nsteps, nsteps, nc, nc)
        Dz = np.repeat(Dz, nsteps, axis=2)  # (nsteps, nsteps, nsteps, nc, nc)
        Dx = Dx.reshape(-1, *Dx.shape[-2:])  # (nsteps**3, nc, nc)
        Dy = Dy.reshape(-1, *Dy.shape[-2:])  # (nsteps**3, nc, nc)
        Dz = Dz.reshape(-1, *Dz.shape[-2:])  # (nsteps**3, nc, nc)
        grog_table = Dx @ Dy @ Dz  # (nsteps**3, nc, nc)

    # perform interpolation
    oshape = np.ceil(np.asarray(oversamp) * np.asarray(shape)).astype(int)
    oshape = tuple([ax.item() for ax in oshape])
    cart_indices = np.ascontiguousarray(indexes[:, -1])
    data = do_interpolation(
        input,
        noncart_indices,
        weights,
        cart_indices,
        bin_starts,
        bin_counts,
        grog_index,
        grog_table,
    )

    # reshape
    data = data.transpose(1, 2, 0)
    if n_batches == 1:
        data = data[0]
    else:
        data = data.reshape(*batch_shape, *data.shape[2:])

    # append stack indexes
    if len(stack_shape) == 0:
        indexes = cart_indices
    elif len(stack_shape) > 1:
        stack_coords = np.unravel_index(indexes[:, 0], stack_shape)
        indexes = np.concatenate((*stack_coords, cart_indices[:, None]), axis=-1)

    return data, indexes, oshape


def _grog_power(G, exponents):
    D, idx = [], 0
    for exp in exponents:
        if np.isclose(exp, 0.0):
            _D = np.eye(G.shape[0], dtype=G.dtype)
        else:
            _D = fmp(G, np.abs(exp)).astype(G.dtype)
            if np.sign(exp) < 0:
                _D = np.linalg.pinv(_D).astype(G.dtype)
        D.append(_D)
        idx += 1

    return np.stack(D, axis=0)


def check_stack(stack_axes):
    if stack_axes is not None:
        _stack_axes = np.sort(np.atleast_1d(stack_axes))
        if _stack_axes.size == 1 and _stack_axes.item() != 0:
            raise ValueError("if we have a single stack axis, it must be the leftmost")
        elif _stack_axes.size > 1:
            if _stack_axes[0] != 0:
                raise ValueError("If provided, stack axis must start from 0")
            _stack_stride = np.unique(np.diff(_stack_axes))
            if _stack_stride.size > 1 or _stack_stride.item() != 1:
                raise ValueError("If provided, stack axes must be contiguous")
        return _stack_axes.tolist()


def kdtree(n_stacks, grid, coords, stack_coords, radius, precision, weighting_mode):
    pfac = 10.0**precision
    stepsize = 10 ** (-precision)
    nsteps = 2 * radius / 10 ** (-precision) + 1
    nsteps = int(nsteps)

    # perorm kd search
    unsorted_indices = _kdtree(grid, coords, radius)

    # flatten object array
    unsorted_indices_val, unsorted_indices_idx = flatten_indices(
        n_stacks, unsorted_indices, stack_coords
    )

    # get the unique bins and the inverse mapping:
    unique_bins, inverse = _unique(unsorted_indices_idx, return_inverse=True)

    # use the inverse mapping to get a sort order that groups identical bins together:
    sort_order = np.argsort(inverse)

    # apply the sort order to both arrays:
    bin_idx = unsorted_indices_idx[sort_order]
    bin_val = unsorted_indices_val[sort_order]

    # now, using _unique on the sorted bin_idx, get the start indices and counts:
    unique_bins, bin_starts, bin_counts = _unique(
        bin_idx, return_index=True, return_counts=True
    )

    # compute distances
    target_coords = grid[np.repeat(unique_bins[:, -1], bin_counts, axis=0), :]
    source_coords = coords[bin_val, :]
    distances = target_coords - source_coords

    # compute weights
    ndim = coords.shape[-1]
    if weighting_mode == "distance":
        weight_scale = ndim**0.5 * radius * 1.00001
        weights = weight_scale - (distances**2).sum(axis=-1) ** 0.5
    elif weighting_mode == "average":
        weights = np.ones(distances.shape[0], dtype=np.float32)

    # compute table index
    tab_idx = (radius + np.round(distances * pfac) / pfac) / stepsize
    tab_idx = np.round(tab_idx).astype(np.float32)
    tab_flattening = np.asarray([1.0, nsteps, nsteps**2], dtype=np.float32)
    tab_idx = tab_idx * tab_flattening[:ndim]
    tab_idx = np.round(tab_idx).astype(int).sum(axis=-1)

    return bin_val, weights, tab_idx, unique_bins, bin_starts, bin_counts


def _unique(arr, return_index=False, return_inverse=False, return_counts=False):
    sorted_idx = np.lexsort(arr.T)
    sorted_arr = arr[sorted_idx]

    unique_mask = np.empty(arr.shape[0], dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = np.any(sorted_arr[1:] != sorted_arr[:-1], axis=1)
    unique_mask_idx = np.where(unique_mask)[0]

    unique_vals = sorted_arr[unique_mask_idx]

    results = [unique_vals]

    if return_index:
        index = sorted_idx[unique_mask_idx]
        results.append(index)

    if return_inverse:
        inverse = np.empty(arr.shape[0], dtype=int)
        inverse[sorted_idx] = np.cumsum(unique_mask) - 1
        results.append(inverse)

    if return_counts:
        counts = np.diff(np.append(unique_mask_idx, arr.shape[0]))
        results.append(counts)

    return tuple(results) if len(results) > 1 else results[0]


def _kdtree(grid, coords, radius):
    kdtree = KDTree(coords)
    return kdtree.query_ball_point(grid, r=radius, workers=-1)


def flatten_indices(n_stacks, indices, stack_coords):
    counts = np.asarray([len(index) for index in indices])

    # find nonzeros
    nonzeros = np.where(counts)[0]
    counts = counts[nonzeros]

    # build
    flattened_indices_val = np.concatenate(indices[nonzeros])
    flattened_indices_idx = np.repeat(nonzeros, counts)
    flattened_indices_idx = np.stack(
        (stack_coords[flattened_indices_val], flattened_indices_idx), axis=-1
    )

    return flattened_indices_val, flattened_indices_idx


def do_interpolation(
    input,
    noncart_indices,
    weights,
    cart_indices,
    bin_starts,
    bin_counts,
    grog_index,
    grog_table,
):
    nbatches = input.shape[1]
    ncoils = input.shape[2]

    # enforce datatype
    input = input.astype(np.complex64)
    noncart_indices = noncart_indices.astype(int)
    weights = weights.astype(np.float32)
    cart_indices = cart_indices.astype(int)
    bin_starts = bin_starts.astype(int)
    bin_counts = bin_counts.astype(int)
    grog_index = grog_index.astype(int)
    grog_table = grog_table.astype(np.complex64)

    # preallocate output
    output = np.zeros((cart_indices.shape[0], nbatches, ncoils), dtype=input.dtype)

    # perform interpolation
    _interpolation(
        output,
        input,
        noncart_indices,
        weights,
        cart_indices,
        bin_starts,
        bin_counts,
        grog_index,
        grog_table,
    )

    return output


@nb.njit(fastmath=True, cache=True, inline="always")  # pragma: no cover
def _matvec(y, A, x):
    ni, nj = A.shape
    for i in range(ni):
        for j in range(nj):
            y[i] += A[i][j] * x[j]


@nb.njit(fastmath=True, cache=True, parallel=True)  # pragma: no cover
def _interpolation(
    output,
    input,
    noncart_indices,
    weights,
    cart_indices,
    bin_starts,
    bin_counts,
    grog_index,
    grog_table,
):
    nsamples, nbatches, ncoils = output.shape

    for n in nb.prange(nsamples):
        bin_start = bin_starts[n]
        bin_count = bin_counts[n]
        total_weight = 0.0

        for b in range(bin_count):
            idx = bin_start + b
            source_index = noncart_indices[idx]

            # get weight
            total_weight += weights[idx]

            # perform interpolation for each element in batch
            for batch in range(nbatches):
                _matvec(
                    output[n, batch],
                    grog_table[grog_index[idx]],
                    weights[idx] * input[source_index, batch],
                )

        # normalize
        output[n] = output[n] / total_weight
