"""GRAPPA operator based interpolation. Adapted for convenience from PyGRAPPA."""

__all__ = ["interp"]

from numpy.typing import ArrayLike

import numpy as np
import numba as nb

from mrinufft._array_compat import with_numpy_cupy, CUPY_AVAILABLE

from .._sigpy import get_device
from .._utils import rescale_coords


@with_numpy_cupy
def interp(
    interpolator: dict,
    input: ArrayLike,
    coords: ArrayLike,
    shape: ArrayLike,
    threadsperblock: int = 128,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    GRAPPA Operator Gridding (GROP) interpolation of Non-Cartesian datasets.

    Parameters
    ----------
    interpolator: dict
        Trained GROG interpolator.
    input : ArrayLike
        Input Non-Cartesian kspace.
    coords : ArrayLike
        Fourier domain coordinates array of shape ``(..., ndims)``.
    shape : ArrayLike[int]
        Cartesian grid size of shape ``(ndim,)``.
        If scalar, isotropic matrix is assumed.
    threadsperblock : int, optional
        Number of CUDA threads per block. The default is ``128``.

    Returns
    -------
    output : ArrayLike
        Output sparse Cartesian kspace.
    indexes : np.ndarray | torch.Tensor
        Sampled k-space points indexes.
    weights : np.ndarray | torch.Tensor
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
    device = get_device(input)
    xp = device.xp

    # get number of spatial dims
    ndim = coords.shape[-1]

    # get batch shape and signal shape
    signal_shape = coords.shape[:-1]
    batch_shape = input.shape[: -len(signal_shape) - 1]

    # reshape data and coordinates
    n_coils = input.shape[-len(signal_shape) - 1]
    n_batchs = int(np.prod(batch_shape))
    input = input.reshape(n_batchs, n_coils, -1)
    coords = coords.reshape(-1, ndim)

    # transpose to (n_samples, n_batches, n_coils)
    input = input.transpose(2, 0, 1)
    input = xp.ascontiguousarray(input)

    # build G
    nsteps = interpolator["x"].shape[0]
    if ndim == 2:
        Gx, Gy = interpolator["x"], interpolator["y"]
        Gx = Gx[None, ...]
        Gy = Gy[:, None, ...]
        Gx = np.repeat(Gx, nsteps, axis=0)  # (nsteps, nsteps, nc, nc)
        Gy = np.repeat(Gy, nsteps, axis=1)  # (nsteps, nsteps, nc, nc)
        Gx = Gx.reshape(-1, *Gx.shape[-2:])  # (nsteps**2, nc, nc)
        Gy = Gy.reshape(-1, *Gy.shape[-2:])  # (nsteps**2, nc, nc)
        G = Gx @ Gy  # (nsteps**2, nc, nc)
    elif ndim == 3:
        Gx, Gy, Gz = interpolator["x"], interpolator["y"], interpolator["z"]
        Gx = Gx[None, None, ...]
        Gy = Gy[None, :, None, ...]
        Gz = Gz[:, None, None, ...]
        Gx = np.repeat(Gx, nsteps, axis=0)  # (nsteps, nsteps, nsteps, nc, nc)
        Gx = np.repeat(Gx, nsteps, axis=1)  # (nsteps, nsteps, nsteps, nc, nc)
        Gy = np.repeat(Gy, nsteps, axis=0)  # (nsteps, nsteps, nsteps, nc, nc)
        Gy = np.repeat(Gy, nsteps, axis=2)  # (nsteps, nsteps, nsteps, nc, nc)
        Gz = np.repeat(Gz, nsteps, axis=1)  # (nsteps, nsteps, nsteps, nc, nc)
        Gz = np.repeat(Gz, nsteps, axis=2)  # (nsteps, nsteps, nsteps, nc, nc)
        Gx = Gx.reshape(-1, *Gx.shape[-2:])  # (nsteps**3, nc, nc)
        Gy = Gy.reshape(-1, *Gy.shape[-2:])  # (nsteps**3, nc, nc)
        Gz = Gz.reshape(-1, *Gz.shape[-2:])  # (nsteps**3, nc, nc)
        G = Gx @ Gy @ Gz  # (nsteps**3, nc, nc)

    # build indexes
    coords = rescale_coords(coords, shape)
    indexes = xp.round(coords)

    with device:
        lut = indexes - coords
        lut = xp.floor(10 * lut, dtype=xp.float32) + nsteps // 2
        lut = lut * xp.asarray([1, nsteps, nsteps**2], dtype=xp.float32)[:ndim]
        lut = lut.sum(axis=-1).astype(int)

        # perform interpolation
        if device.id < 0:
            output = do_interpolation(input, G, lut)
        else:
            output = do_interpolation_cuda(input, G, lut, threadsperblock)

        # finalize indexes
        if xp.isscalar(shape):
            shape = [shape] * ndim
        else:
            shape = list(shape)[-ndim:]
            shape = shape[::-1]
        indexes = indexes + xp.asarray(shape, dtype=indexes.dtype) // 2
        indexes = indexes.astype(int)

        # ravel indexes
        unfolding = xp.cumprod([1] + shape[: ndim - 1])
        flattened_idx = xp.asarray(unfolding, dtype=indexes.dtype) * indexes
        flattened_idx = flattened_idx.sum(axis=-1)

    # count elements
    _, idx, counts = xp.unique(flattened_idx, return_inverse=True, return_counts=True)

    # compute weights
    weights = counts[idx].astype(xp.float32)
    weights = 1 / weights

    # reformat shape
    output = output.transpose(1, 2, 0)  # (n_batches, n_coils, n_samples)
    output = output.reshape(*batch_shape, n_coils, *signal_shape)
    indexes = indexes.reshape(*signal_shape, ndim)
    weights = weights.reshape(*signal_shape)

    # remove out-of-boundaries
    for n in range(ndim):
        # below minimum
        outside = indexes[..., n] < 0
        output[..., outside] = 0.0
        indexes[..., n][outside] = 0  # collapse to minimum
        # weights[outside] = 0.0

        # above maximum
        outside = indexes[..., n] >= shape[n]
        output[..., outside] = 0.0
        indexes[..., n][outside] = shape[n] - 1  # collapse to maxmum
        # weights[outside] = 0.0

    # enforce contiguity
    output = xp.ascontiguousarray(output)
    indexes = xp.ascontiguousarray(indexes)
    weights = xp.ascontiguousarray(weights)

    return output, indexes, weights


# %% subroutines
def do_interpolation(noncart, G, lut):
    cart = np.zeros(noncart.shape, dtype=noncart.dtype)
    _interp(cart, noncart, G, lut)
    return cart


@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _dot_product(out, in_a, in_b):
    row, col = in_b.shape

    for i in range(row):
        for j in range(col):
            out[j] += in_b[i][j] * in_a[j]

    return out


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _interp(data_out, data_in, interp, lut):
    # get data dimension
    nsamples, batch_size, _ = data_in.shape

    for i in nb.prange(nsamples * batch_size):
        sample = i // batch_size
        batch = i % batch_size
        idx = lut[sample]

        _dot_product(data_out[sample][batch], data_in[sample][batch], interp[idx])


# %% CUDA
if CUPY_AVAILABLE:
    from numba import cuda

    def do_interpolation_cuda(noncart, G, lut, threadsperblock):
        blockspergrid = (
            np.prod(noncart.shape[-2:]) + (threadsperblock - 1)
        ) // threadsperblock
        blockspergrid = int(blockspergrid)

        with get_device(noncart) as device:
            cart = device.xp.zeros(noncart.shape, dtype=noncart.dtype)
            _interp_cuda[blockspergrid, threadsperblock](cart, noncart, G, lut)

        return cart

    @cuda.jit(device=True, inline=True)  # pragma: no cover
    def _dot_product_cuda(out, in_a, in_b):
        row, col = in_b.shape

        for i in range(row):
            for j in range(col):
                out[j] += in_b[i][j] * in_a[j]

        return out

    @cuda.jit(fastmath=True)  # pragma: no cover
    def _interp_cuda(data_out, data_in, interp, lut):
        # get data dimension
        nvoxels, batch_size, _ = data_in.shape

        i = nb.cuda.grid(1)
        if i < nvoxels * batch_size:
            sample = i // batch_size
            batch = i % batch_size
            idx = lut[sample]

            _dot_product_cuda(
                data_out[sample][batch], data_in[sample][batch], interp[idx]
            )
