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
) -> tuple[NDArray[complex], NDArray[int], NDArray[float]]:
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
        Output sparse Cartesian kspace.
    indexes : NDArray[int]
        Sampled k-space points indexes.
    weights : NDArray[float]
        Number of occurrences of each k-space sample.

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
    nsteps = 1.0 / 10 ** (-precision) + 1
    nsteps = int(nsteps)

    # compute exponends
    deltas = (np.arange(nsteps) - (nsteps - 1) // 2) / (nsteps - 1)

    # pre-compute partial operators
    Dx = _grog_power(interpolator["x"], deltas).astype(input.dtype)  # (nsteps, nc, nc)
    Dy = _grog_power(interpolator["y"], deltas).astype(input.dtyp)  # (nsteps, nc, nc)
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
        n_batchs = int(np.prod(batch_shape[:-1]))
        n_stacks = int(np.prod(stack_shape))
        n_samples = int(np.prod(signal_shape))
        n_coils = batch_shape[-1]

        # reshape data to (nsamples, nbatchs, ncoils)
        input = input.reshape(n_batchs, n_coils, -1)
        input = input.transpose(2, 0, 1)
        input = xp.ascontiguousarray(input)

        # generate stack coordinate
        stack_coords = xp.meshgrid(
            *[xp.arange(0, stack_size) for stack_size in stack_shape],
            indexing="ij",
        )
        stack_coords = xp.stack([ax.ravel() for ax in stack_coords], axis=-1)
        stack_coords = stack_coords * xp.asarray(stack_shape[1:] + (1,))
        stack_coords = stack_coords.sum(axis=-1)
        stack_coords = xp.repeat(stack_coords, n_samples)

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
        indices = kdtree(grid, coords, radius)

        # divide amongst stacks
        noncart_indices, cart_indices = sort_target(n_stacks, indices, stack_coords)

        # power
        Dx = xp.asarray(Dx)
        Dy = xp.asarray(Dy)
        if Dz is not None:
            Dz = xp.asarray(Dz)

        # perform interpolation
        oshape = xp.ceil(oversamp * xp.asarray(shape)).astype(int)
        oshape = tuple([ax.item() for ax in oshape])
        data, indexes = do_interpolation(
            input,
            coords,
            grid,
            cart_indices,
            noncart_indices,
            Dx,
            Dy,
            Dz,
            precision,
            radius,
            threadsperblock,
        )

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


def kdtree(grid, coords, radius):
    device = get_device(grid)
    if device.id < 0 or CUPY_POST_140 is False:
        return _kdtree_cpu(grid, coords, radius).tolist()
    else:
        return _kdtree_cuda(grid, coords, radius).tolist()


@with_numpy
def _kdtree_cpu(grid, coords, radius):
    kdtree = KDTree(coords)
    return kdtree.query_ball_point(grid, r=radius, workers=-1)


def _kdtree_cuda(grid, coords, radius):
    kdtree = cuKDTree(coords)
    return kdtree.query_ball_point(grid, r=radius)


def sort_target(n_stacks, indices, stack_coords):
    if CUPY_POST_140:
        return _sort_target(n_stacks, indices, stack_coords)
    else:
        return _sort_target_cpu(n_stacks, indices, stack_coords)


def _sort_target(n_stacks, indices, stack_coords):
    device = get_device(indices)
    xp = device.xp
    with device:
        counts = xp.asarray([len(index) for index in indices])

        # find nonzeros
        nonzeros = xp.where(counts)[0]
        counts = counts[nonzeros]

        # build
        noncart = xp.concatenate(indices[nonzeros])
        cart = xp.repeat(nonzeros, counts)
        cart = xp.stack((stack_coords[noncart], cart), axis=-1)

        return noncart, cart


@with_numpy
def _sort_target_cpu(n_stacks, indices, stack_coords):
    return _sort_target(n_stacks, indices, stack_coords)


def do_interpolation(
    input,
    coords,
    grid,
    cart_indices,
    noncart_indices,
    Dx,
    Dy,
    Dz,
    precision,
    radius,
    threadsperblock,
):
    ndim = coords.shape[-1]
    device = get_device(input)
    xp = device.xp

    with device:
        unique_cart_indices, indices_map = xp.unique(
            cart_indices, axis=0, return_inverse=True
        )
        output = xp.zeros(unique_cart_indices.shape[0], dtype=input.dtype)
        norm = xp.zeros(unique_cart_indices.shape[0], dtype=input.dtype)

        if device.id < 0:
            if ndim == 2:
                _interpolation_2D(
                    output,
                    norm,
                    indices_map,
                    cart_indices,
                    noncart_indices,
                    coords,
                    grid,
                    Dx,
                    Dy,
                    precision,
                    radius,
                )
            elif ndim == 3:
                _interpolation_3D(
                    output,
                    norm,
                    indices_map,
                    cart_indices,
                    noncart_indices,
                    coords,
                    grid,
                    Dx,
                    Dy,
                    Dz,
                    precision,
                    radius,
                )
        # else:
        #     blockspergrid = (output.shape[0] + (threadsperblock - 1)) // threadsperblock
        #     blockspergrid = int(blockspergrid)
        #     if ndim == 2:
        #         _cu_interpolation_2D[blockspergrid, threadsperblock](
        #             output,
        #             norm,
        #             indices_map,
        #             cart_indices,
        #             noncart_indices,
        #             coords,
        #             grid,
        #             Dx,
        #             Dy,
        #             precision,
        #             radius,
        #         )
        #     elif ndim == 3:
        #         [blockspergrid, threadsperblock](
        #             output,
        #             norm,
        #             indices_map,
        #             cart_indices,
        #             noncart_indices,
        #             coords,
        #             grid,
        #             Dx,
        #             Dy,
        #             Dz,
        #             precision,
        #             radius,
        #         )

    return output / norm


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

    return y


@nb.njit(fastmath=True)  # pragma: no cover
def _interpolation_2D(
    output,
    norm,
    indices_map,
    cart_indices,
    input,
    noncart_indices,
    coords,
    grid,
    Dx,
    Dy,
    precision,
    radius,
):
    nsamples = indices_map.shape[0]
    _, nbatches, ncoils = input.shape

    pfac = 10.0**precision
    G = np.zeros((ncoils, ncoils), dtype=input.dtype)

    for n in range(nsamples):
        idx = indices_map[n]
        stack_idx, grid_idx = cart_indices[idx]
        coord_idx = noncart_indices[n]

        # select source and target coordinates
        source_coord = coords[coord_idx]
        target_coord = grid[grid_idx]

        # compute distance
        dx = source_coord[0] - target_coord[0]
        dy = source_coord[1] - target_coord[1]

        # find index in interpolator
        nx = int(round(dx * pfac) / pfac + radius)
        ny = int(round(dy * pfac) / pfac + radius)

        # compute interpolator
        _matmul(Dx[nx], Dy[ny], 0 * G)

        # update count
        weight = 1.0  # or 1.0 / (1.0 + (dx**2 + dy**2)**0.5)
        norm[idx] += weight

        # perform interpolation for each element in batch
        for batch in range(nbatches):
            _matvec(G, weight * input[coord_idx, batch], output[idx, batch])


@nb.njit(fastmath=True)  # pragma: no cover
def _interpolation_3D(
    output,
    norm,
    indices_map,
    cart_indices,
    input,
    noncart_indices,
    coords,
    grid,
    Dx,
    Dy,
    Dz,
    precision,
    radius,
):
    nsamples = indices_map.shape[0]
    _, nbatches, ncoils = input.shape

    pfac = 10.0**precision
    _G = np.zeros((ncoils, ncoils), dtype=input.dtype)
    G = np.zeros((ncoils, ncoils), dtype=input.dtype)

    for n in range(nsamples):
        idx = indices_map[n]
        stack_idx, grid_idx = cart_indices[idx]
        coord_idx = noncart_indices[n]

        # select source and target coordinates
        source_coord = coords[coord_idx]
        target_coord = grid[grid_idx]

        # compute distance
        dx = source_coord[0] - target_coord[0]
        dy = source_coord[1] - target_coord[1]
        dz = source_coord[2] - target_coord[2]

        # find index in interpolator
        nx = int(round(dx * pfac) / pfac + radius)
        ny = int(round(dy * pfac) / pfac + radius)
        nz = int(round(dz * pfac) / pfac + radius)

        # compute interpolator
        _matmul(Dx[nx], Dy[ny], 0 * _G)
        _matmul(_G, Dz[nz], 0 * G)

        # update count
        weight = 1.0  # or 1.0 / (1.0 + (dx**2 + dy**2)**0.5)
        norm[idx] += weight

        # perform interpolation for each element in batch
        for batch in range(nbatches):
            _matvec(G, weight * input[coord_idx, batch], output[idx, batch])


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

        return y

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interpolation_2D(
        output,
        norm,
        indices_map,
        cart_indices,
        input,
        noncart_indices,
        coords,
        grid,
        Dx,
        Dy,
        precision,
        radius,
    ):
        nsamples = indices_map.shape[0]
        _, nbatches, ncoils = input.shape

        pfac = 10.0**precision
        G = np.zeros((ncoils, ncoils), dtype=input.dtype)

        for n in range(nsamples):
            idx = indices_map[n]
            stack_idx, grid_idx = cart_indices[idx]
            coord_idx = noncart_indices[n]

            # select source and target coordinates
            source_coord = coords[coord_idx]
            target_coord = grid[grid_idx]

            # compute distance
            dx = source_coord[0] - target_coord[0]
            dy = source_coord[1] - target_coord[1]

            # find index in interpolator
            nx = int(round(dx * pfac) / pfac + radius)
            ny = int(round(dy * pfac) / pfac + radius)

            # compute interpolator
            _matmul(Dx[nx], Dy[ny], 0 * G)

            # update count
            weight = 1.0  # or 1.0 / (1.0 + (dx**2 + dy**2)**0.5)
            norm[idx] += weight

            # perform interpolation for each element in batch
            for batch in range(nbatches):
                _matvec(G, weight * input[coord_idx, batch], output[idx, batch])

    @nb.njit(fastmath=True)  # pragma: no cover
    def _interpolation_3D(
        output,
        norm,
        indices_map,
        cart_indices,
        input,
        noncart_indices,
        coords,
        grid,
        Dx,
        Dy,
        Dz,
        precision,
        radius,
    ):
        nsamples = indices_map.shape[0]
        _, nbatches, ncoils = input.shape

        pfac = 10.0**precision
        _G = np.zeros((ncoils, ncoils), dtype=input.dtype)
        G = np.zeros((ncoils, ncoils), dtype=input.dtype)

        for n in range(nsamples):
            idx = indices_map[n]
            stack_idx, grid_idx = cart_indices[idx]
            coord_idx = noncart_indices[n]

            # select source and target coordinates
            source_coord = coords[coord_idx]
            target_coord = grid[grid_idx]

            # compute distance
            dx = source_coord[0] - target_coord[0]
            dy = source_coord[1] - target_coord[1]
            dz = source_coord[2] - target_coord[2]

            # find index in interpolator
            nx = int(round(dx * pfac) / pfac + radius)
            ny = int(round(dy * pfac) / pfac + radius)
            nz = int(round(dz * pfac) / pfac + radius)

            # compute interpolator
            _matmul(Dx[nx], Dy[ny], 0 * _G)
            _matmul(_G, Dz[nz], 0 * G)

            # update count
            weight = 1.0  # or 1.0 / (1.0 + (dx**2 + dy**2)**0.5)
            norm[idx] += weight

            # perform interpolation for each element in batch
            for batch in range(nbatches):
                _matvec(G, weight * input[coord_idx, batch], output[idx, batch])
