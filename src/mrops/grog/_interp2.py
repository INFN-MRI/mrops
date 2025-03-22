"""GRAPPA operator based interpolation. Adapted for convenience from PyGRAPPA."""

__all__ = ["interp"]

from numpy.typing import NDArray

import numpy as np
import numba as nb

from scipy.linalg import fractional_matrix_power as fmp
from scipy.spatial import KDTree

from mrinufft._array_compat import with_numpy_cupy, with_numpy, CUPY_AVAILABLE

from .._sigpy import get_device
from .._utils import rescale_coords

if CUPY_AVAILABLE:
    try:
        from cupyx.scipy.spatial import KDTree as cuKDTree

        CUPY_POST_140 = True
    except Exception:
        CUPY_POST_140 = False


@with_numpy_cupy
def interp(
    interpolator: dict,
    input: NDArray[complex],
    coords: NDArray[float],
    shape: list[int] | tuple[int],
    stack_axes: list[int] | tuple[int] | None = None,
    oversamp: float = 1.0,
    radius: float = 0.75,
    precision: int = 1,
    threadsperblock: int = 128,
) -> tuple[NDArray[complex], NDArray[int], tuple[int]]:
    """
    GRAPPA Operator Gridding (GROP) interpolation of Non-Cartesian datasets.

    Parameters
    ----------
    interpolator: dict
        Trained GROG interpolator.
    input : NDArray[complex]
        Input Non-Cartesian kspace.
    coords : NDArray[float]
        Fourier domain coordinates array of shape ``(..., ndims)``.
    shape : list[int] | tuple[int]
        Cartesian grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    threadsperblock : int, optional
        Number of CUDA threads per block. The default is ``128``.

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
    stack_axes = check_stack(stack_axes)
    device = get_device(input)
    xp = device.xp

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
            input.dtyp
        )  # (nsteps, nc, nc), 3D only
    else:
        Dz = None

    with device:
        if stack_axes is None:
            stack_shape = ()
        else:
            stack_shape = coords.shape[: len(stack_axes)]
        signal_shape = coords.shape[len(stack_shape) : -1]
        ndim = coords.shape[-1]

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
        input = xp.ascontiguousarray(input)

        # generate stack coordinate
        if len(stack_shape) > 0:
            stack_coords = xp.meshgrid(
                *[xp.arange(0, stack_size) for stack_size in stack_shape],
                indexing="ij",
            )
            stack_coords = xp.stack([ax.ravel() for ax in stack_coords], axis=-1)
            stack_coords = stack_coords * xp.asarray(stack_shape[1:] + (1,))
            stack_coords = stack_coords.sum(axis=-1)
            stack_coords = xp.repeat(stack_coords, n_samples)
            stack_coords = stack_coords.astype(int)
        else:
            stack_coords = xp.zeros(n_samples, dtype=int)

        # reshape coordinates
        coords = coords.reshape(-1, ndim)
        coords = rescale_coords(coords, shape)

        # build target
        grid = xp.meshgrid(
            *[
                xp.linspace(
                    -shape[n] // 2, shape[n] // 2 - 1, int(np.ceil(oversamp * shape[n]))
                )
                for n in range(ndim)
            ],
            indexing="ij",
        )
        grid = xp.stack([ax.ravel() for ax in grid], axis=-1).astype(xp.float32)

        # remove indices outside support
        # inside = (grid**2).sum(axis=-1)**0.5 <= 0.5 * xp.max(shape)
        # grid = xp.ascontiguousarray(grid[inside, :])

        # find source for each target
        noncart_indices, indexes, bin_starts, bin_counts = kdtree(
            n_stacks, grid, coords, stack_coords, radius
        )

        # power
        Dx = xp.asarray(Dx)
        Dy = xp.asarray(Dy)
        if Dz is not None:
            Dz = xp.asarray(Dz)

        # perform interpolation
        oshape = xp.ceil(oversamp * xp.asarray(shape)).astype(int)
        oshape = tuple([ax.item() for ax in oshape])
        cart_indices = xp.ascontiguousarray(indexes[:, -1])
        data = do_interpolation(
            input,
            coords,
            grid,
            cart_indices,
            bin_starts,
            bin_counts,
            noncart_indices,
            Dx,
            Dy,
            Dz,
            precision,
            radius,
            threadsperblock,
        )

    # reshape
    data = data.transpose(1, 2, 0)
    if n_batches == 1:
        data = data[0]
    else:
        data = data.reshape(*batch_shape, *data.shape[2:])

    return data, indexes, oshape


# %% subroutines
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


def kdtree(n_stacks, grid, coords, stack_coords, radius):
    device = get_device(grid)
    xp = device.xp

    if device.id < 0 or CUPY_POST_140 is False:
        unsorted_indices = _kdtree_cpu(grid, coords, radius)
    else:
        unsorted_indices = _kdtree_cuda(grid, coords, radius)

    # flatten unsorted indices
    unsorted_indices_val, unsorted_indices_idx = flatten_indices(
        n_stacks, unsorted_indices, stack_coords
    )

    # Get the unique bins and the inverse mapping:
    unique_bins, inverse = xp.unique(unsorted_indices_idx, axis=0, return_inverse=True)

    # Use the inverse mapping to get a sort order that groups identical bins together:
    sort_order = xp.argsort(inverse)

    # Apply the sort order to both arrays:
    bin_idx = unsorted_indices_idx[sort_order]
    bin_val = unsorted_indices_val[sort_order]

    # (Optional) Now, using xp.unique on the sorted bin_idx, get the start indices and counts:
    unique_bins, bin_starts, bin_counts = xp.unique(
        bin_idx, axis=0, return_index=True, return_counts=True
    )

    return bin_val, unique_bins, bin_starts, bin_counts


@with_numpy
def _kdtree_cpu(grid, coords, radius):
    kdtree = KDTree(coords)
    return kdtree.query_ball_point(grid, r=radius, workers=-1)


def _kdtree_cuda(grid, coords, radius):
    kdtree = cuKDTree(coords)
    return kdtree.query_ball_point(grid, r=radius)


def flatten_indices(n_stacks, indices, stack_coords):
    if CUPY_POST_140:
        return _flatten_indices(n_stacks, indices, stack_coords)
    else:
        return _flatten_indices_cpu(n_stacks, indices, stack_coords)


def _flatten_indices(n_stacks, indices, stack_coords):
    device = get_device(indices)
    xp = device.xp
    with device:
        counts = xp.asarray([len(index) for index in indices])

        # find nonzeros
        nonzeros = xp.where(counts)[0]
        counts = counts[nonzeros]

        # build
        flattened_indices_val = xp.concatenate(indices[nonzeros])
        flattened_indices_idx = xp.repeat(nonzeros, counts)
        flattened_indices_idx = xp.stack(
            (stack_coords[flattened_indices_val], flattened_indices_idx), axis=-1
        )

        return flattened_indices_val, flattened_indices_idx


@with_numpy
def _flatten_indices_cpu(n_stacks, indices, stack_coords):
    return _flatten_indices(n_stacks, indices, stack_coords)


def do_interpolation(
    input,
    coords,
    grid,
    cart_indices,
    bin_starts,
    bin_counts,
    noncart_indices,
    Dx,
    Dy,
    Dz,
    precision,
    radius,
    threadsperblock,
):
    device = get_device(input)
    xp = device.xp

    # get dimensions
    ndim = coords.shape[-1]
    nbatches = input.shape[1]
    ncoils = input.shape[2]

    # enforce datatype
    stepsize = 10 ** (-precision)
    input = input.astype(xp.complex64)
    coords = coords.astype(xp.float32)
    grid = grid.astype(xp.float32)
    cart_indices = cart_indices.astype(int)
    bin_starts = bin_starts.astype(int)
    bin_counts = bin_counts.astype(int)
    noncart_indices = noncart_indices.astype(int)
    Dx = Dx.astype(xp.complex64)
    Dy = Dy.astype(xp.complex64)
    if ndim == 3:
        Dz = Dz.astype(xp.complex64)

    with device:
        output = xp.zeros((cart_indices.shape[0], nbatches, ncoils), dtype=input.dtype)
        if device.id < 0:
            if ndim == 2:
                G = np.zeros((ncoils, ncoils), dtype=input.dtype)
                _interpolation_2D(
                    output,
                    input,
                    cart_indices,
                    bin_starts,
                    bin_counts,
                    noncart_indices,
                    coords,
                    grid,
                    Dx,
                    Dy,
                    precision,
                    stepsize,
                    radius,
                    G,
                )
            elif ndim == 3:
                G = np.zeros((ncoils, ncoils), dtype=input.dtype)
                _G = np.zeros((ncoils, ncoils), dtype=input.dtype)
                _interpolation_3D(
                    output,
                    input,
                    cart_indices,
                    bin_starts,
                    bin_counts,
                    noncart_indices,
                    coords,
                    grid,
                    Dx,
                    Dy,
                    Dz,
                    precision,
                    stepsize,
                    radius,
                    G,
                    _G,
                )
        else:
            blockspergrid = (output.shape[0] + (threadsperblock - 1)) // threadsperblock
            blockspergrid = int(blockspergrid)
            if ndim == 2:
                G = xp.zeros((ncoils, ncoils), dtype=input.dtype)
                _cu_interpolation_2D[blockspergrid, threadsperblock](
                    output,
                    input,
                    cart_indices,
                    bin_starts,
                    bin_counts,
                    noncart_indices,
                    coords,
                    grid,
                    Dx,
                    Dy,
                    precision,
                    stepsize,
                    radius,
                    G,
                )
            elif ndim == 3:
                G = xp.zeros((ncoils, ncoils), dtype=input.dtype)
                _G = xp.zeros((ncoils, ncoils), dtype=input.dtype)
                _cu_interpolation_3D[blockspergrid, threadsperblock](
                    output,
                    input,
                    cart_indices,
                    bin_starts,
                    bin_counts,
                    noncart_indices,
                    coords,
                    grid,
                    Dx,
                    Dy,
                    Dz,
                    precision,
                    stepsize,
                    radius,
                    G,
                    _G,
                )

    return output


@nb.njit(fastmath=True, cache=True, inline="always")  # pragma: no cover
def _matmul(A, B, C):
    ni, nk = A.shape
    nj = B.shape[1]
    for i in range(ni):
        for j in range(nj):
            C[i, j] = 0.0
            for k in range(nk):
                C[i, j] += A[i, k] * B[k, j]


@nb.njit(fastmath=True, cache=True, inline="always")  # pragma: no cover
def _matvec(A, x, y):
    ni, nj = A.shape
    for i in range(ni):
        for j in range(nj):
            y[i] += A[i][j] * x[j]


@nb.njit(fastmath=True, cache=True, parallel=True)  # pragma: no cover
def _interpolation_2D(
    output,
    input,
    cart_indices,
    bin_starts,
    bin_counts,
    noncart_indices,
    coords,
    grid,
    Dx,
    Dy,
    precision,
    stepsize,
    radius,
    G,
):
    nsamples, nbatches, ncoils = output.shape
    pfac = 10.0**precision

    for n in range(nsamples):
        target_index = cart_indices[n]
        target_coord = grid[target_index]
        bin_start = bin_starts[n]
        bin_count = bin_counts[n]

        for b in range(bin_count):
            G[:] = 0.0  # reset interpolator
            idx = bin_start + b
            source_index = noncart_indices[idx]
            source_coord = coords[source_index]

            # compute distance
            dx = target_coord[0] - source_coord[0]
            dy = target_coord[1] - source_coord[1]

            # find index in interpolator
            nx = int((radius + round(dx * pfac) / pfac) / stepsize)
            ny = int((radius + round(dy * pfac) / pfac) / stepsize)

            # compute interpolator
            _matmul(Dx[nx], Dy[ny], G)

            # perform interpolation for each element in batch
            for batch in range(nbatches):
                _matvec(G, input[source_index, batch], output[n, batch])

        # normalize
        output[n] = output[n] / bin_count


@nb.njit(fastmath=True, cache=True, parallel=True)  # pragma: no cover
def _interpolation_3D(
    output,
    input,
    cart_indices,
    bin_starts,
    bin_counts,
    noncart_indices,
    coords,
    grid,
    Dx,
    Dy,
    Dz,
    precision,
    stepsize,
    radius,
    G,
    _G,
):
    nsamples, nbatches, ncoils = output.shape
    pfac = 10.0**precision

    for n in nb.prange(nsamples):
        target_index = cart_indices[n]
        target_coord = grid[target_index]
        bin_start = bin_starts[n]
        bin_count = bin_counts[n]

        for b in range(bin_count):
            G[:] = 0.0  # reset interpolator
            _G[:] = 0.0  # reset interpolator
            idx = bin_start + b
            source_index = noncart_indices[idx]
            source_coord = coords[source_index]

            # compute distance
            dx = target_coord[0] - source_coord[0]
            dy = target_coord[1] - source_coord[1]
            dz = target_coord[2] - source_coord[2]

            # find index in interpolator
            nx = int((radius + round(dx * pfac) / pfac) / stepsize)
            ny = int((radius + round(dy * pfac) / pfac) / stepsize)
            nz = int((radius + round(dz * pfac) / pfac) / stepsize)

            # compute interpolator
            _matmul(Dx[nx], Dy[ny], _G)
            _matmul(_G, Dz[nz], G)

            # perform interpolation for each element in batch
            for batch in range(nbatches):
                _matvec(G, input[source_index, batch], output[n, batch])

        # normalize
        output[n] = output[n] / bin_count


# %% CUDA
if CUPY_AVAILABLE:
    from numba import cuda

    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _cumatmul(A, B, C):
        ni, nk = A.shape
        nj = B.shape[1]
        for i in range(ni):
            for j in range(nj):
                C[i, j] = 0.0
                for k in range(nk):
                    C[i, j] += A[i, k] * B[k, j]

    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _cumatvec(A, x, y):
        ni, nj = A.shape
        for i in range(ni):
            for j in range(nj):
                y[i] += A[i][j] * x[j]

    @cuda.jit(fastmath=True, cache=True)  # pragma: no cover
    def _cu_interpolation_2D(
        output,
        input,
        cart_indices,
        bin_starts,
        bin_counts,
        noncart_indices,
        coords,
        grid,
        Dx,
        Dy,
        precision,
        stepsize,
        radius,
        G,
    ):
        nsamples, nbatches, ncoils = output.shape
        pfac = 10.0**precision

        n = cuda.grid(1)
        if n < nsamples:
            target_index = cart_indices[n]
            target_coord = grid[target_index]
            bin_start = bin_starts[n]
            bin_count = bin_counts[n]

            for b in range(bin_count):
                G[:] = 0.0  # reset interpolator
                idx = bin_start + b
                source_index = noncart_indices[idx]
                source_coord = coords[source_index]

                # compute distance
                dx = target_coord[0] - source_coord[0]
                dy = target_coord[1] - source_coord[1]

                # find index in interpolator
                nx = int((radius + round(dx * pfac) / pfac) / stepsize)
                ny = int((radius + round(dy * pfac) / pfac) / stepsize)

                # compute interpolator
                _cumatmul(Dx[nx], Dy[ny], 0.0 * G)

                # perform interpolation for each element in batch
                for batch in range(nbatches):
                    _cumatvec(
                        G, input[source_index, batch], output[target_index, batch]
                    )

            # normalize
            output[target_index] = output[target_index] / bin_count

    @cuda.jit(fastmath=True, cache=True)  # pragma: no cover
    def _cu_interpolation_3D(
        output,
        input,
        cart_indices,
        bin_starts,
        bin_counts,
        noncart_indices,
        coords,
        grid,
        Dx,
        Dy,
        Dz,
        precision,
        stepsize,
        radius,
        G,
        _G,
    ):
        nsamples, nbatches, ncoils = output.shape
        pfac = 10.0**precision

        n = cuda.grid(1)
        if n < nsamples:
            target_index = cart_indices[n]
            target_coord = grid[target_index]
            bin_start = bin_starts[n]
            bin_count = bin_counts[n]

            for b in range(bin_count):
                G[:] = 0.0  # reset interpolator
                _G[:] = 0.0  # reset interpolator
                idx = bin_start + b
                source_index = noncart_indices[idx]
                source_coord = coords[source_index]

                # compute distance
                dx = target_coord[0] - source_coord[0]
                dy = target_coord[1] - source_coord[1]
                dz = target_coord[2] - source_coord[2]

                # find index in interpolator
                nx = int((radius + round(dx * pfac) / pfac) / stepsize)
                ny = int((radius + round(dy * pfac) / pfac) / stepsize)
                nz = int((radius + round(dz * pfac) / pfac) / stepsize)

                # compute interpolator
                _cumatmul(Dx[nx], Dy[ny], 0.0 * _G)
                _cumatmul(_G, Dz[nz], 0.0 * G)

                # perform interpolation for each element in batch
                for batch in range(nbatches):
                    _cumatvec(G, input[source_index, batch], output[n, batch])

            # normalize
            output[n] = output[n] / bin_count
