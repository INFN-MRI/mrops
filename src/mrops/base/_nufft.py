"""Non Uniform Fast Fourier Transform."""

__all__ = ["nufft", "nufft_adjoint"]

import gc
import math

from types import SimpleNamespace

import numpy as np
from numpy.typing import ArrayLike

import mrinufft

from mrinufft._array_compat import with_numpy_cupy
from mrinufft.operators.interfaces.utils import is_cuda_array

from .._sigpy.fourier import estimate_shape
from .._utils import rescale_coords

if mrinufft.check_backend("cufinufft"):
    import cupy as cp


def nufft(
    input: ArrayLike,
    coords: ArrayLike,
    oversamp: float = 1.25,
    eps: float = 1e-3,
    normalize_coords: bool = True,
) -> ArrayLike:
    """
    Non-uniform Fast Fourier Transform.

    Parameters
    ----------
    input : ArrayLike
        Input signal domain array of shape
        ``(..., n_{ndim - 1}, ..., n_1, n_0)``,
        where ``ndim`` is specified by ``coord.shape[-1]``. The nufft
        is applied on the last ``ndim axes``, and looped over
        the remaining axes.
    coords : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coords : bool, optional
        Normalize coordinates between -pi and pi. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    Returns
    -------
    ArrayLike
        Fourier domain data of shape
        ``input.shape[:-ndim] + coords.shape[:-1]``.

    """
    ndim = coords.shape[-1]
    ishape = input.shape[-ndim:]
    plan = __nufft_init__(coords, ishape, oversamp, eps, normalize_coords)
    output = _apply(plan, input)
    return output.reshape(*output.shape[:-1], *coords.shape[:-1])


def nufft_adjoint(
    input: ArrayLike,
    coords: ArrayLike,
    oshape: ArrayLike | None = None,
    oversamp: float = 1.25,
    eps: float = 1e-3,
    normalize_coords: bool = True,
) -> ArrayLike:
    """
    Adjoint non-uniform Fast Fourier Transform.

    Parameters
    ----------
    input : ArrayLike
        Input Fourier domain array of shape
        ``(..., n_{ndim - 1}, ..., n_1, n_0)``,
        where ``ndim`` is specified by ``coord.shape[-1]``. The nufft
        is applied on the last ``ndim axes``, and looped over
        the remaining axes.
    coord : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    oshape : ArrayLike[int] | None, optional
        Output shape of the form ``(..., n_{ndim - 1}, ..., n_1, n_0)``.
        The default is ``None`` (estimated from ``coord``).
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coords : bool, optional
        Normalize coordinates between -pi and pi. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    Returns
    -------
    ArrayLike
        Signal domain data of shape
        ``input.shape[:-ndim] + coords.shape[:-1]``.

    """
    fourier_ndim = len(coords) - 1
    input = input.reshape(*input.shape[:-fourier_ndim], -1)
    plan = __nufft_init__(coords, oshape, oversamp, eps, normalize_coords)
    return _apply_adj(plan, input)


# %% local subroutines
@with_numpy_cupy
def __nufft_init__(
    coords: ArrayLike,
    shape: ArrayLike | None = None,
    oversamp: float = 1.25,
    eps: float = 1e-6,
    normalize_coords: bool = True,
):
    if shape is None:
        shape = estimate_shape(coords)

    # enforce single precision
    coords = coords.astype(np.float32)

    # normalize
    if normalize_coords:
        coords = rescale_coords(coords, 2 * math.pi)

    # enforce numpy array for coords
    try:
        coords = coords.get()
    except Exception:
        pass

    # prepare CPU nufft
    cpu_nufft = mrinufft.get_operator("finufft")(
        samples=coords.reshape(-1, coords.shape[-1]),
        shape=shape[::-1],
        squeeze_dims=True,
        upsampfac=oversamp,
        eps=eps,
    )

    if mrinufft.check_backend("cufinufft"):
        gpu_nufft = mrinufft.get_operator("cufinufft")(
            samples=coords.reshape(-1, coords.shape[-1]),
            shape=shape[::-1],
            squeeze_dims=True,
            upsampfac=oversamp,
            eps=eps,
        )
    else:
        gpu_nufft = None

    return SimpleNamespace(cpu=cpu_nufft, gpu=gpu_nufft)


@with_numpy_cupy
def _apply(plan, input):
    # reshape from (..., *grid_shape) to (B, *grid_shape)
    ndim = plan.cpu.ndim
    broadcast_shape = input.shape[:-ndim]
    input = input.reshape(-1, *input.shape[-ndim:])

    # select operator based on computational device
    if is_cuda_array(input):
        _nufft = plan.gpu
    else:
        _nufft = plan.cpu

    # actual computation
    if input.ndim == ndim:
        output = _nufft.op(input)
    else:
        output = np.stack([_nufft.op(batch) for batch in input])

    # reshape from (B, samples) to (..., samples)
    if output.ndim != 1:
        output = output.reshape(*broadcast_shape, *output.shape[1:])

    # clean-up
    if is_cuda_array(input):
        gc.collect()
        cp._default_memory_pool.free_all_blocks()

    return output


@with_numpy_cupy
def _apply_adj(plan, input):
    # reshape from (..., samples) to (B, samples)
    nsamples = plan.cpu.n_samples
    broadcast_shape = input.shape[:-1]
    input = input.reshape(-1, nsamples)

    # select operator based on computational device
    if is_cuda_array(input):
        _nufft = plan.gpu
    else:
        _nufft = plan.cpu

    # actual computation
    if input.ndim == 1:
        output = _nufft.adj_op(input)
    else:
        output = np.stack([_nufft.adj_op(batch) for batch in input])

    # reshape from (B, *grid_shape) to (..., *grid_shape)
    if input.ndim != 1:
        output = output.reshape(*broadcast_shape, *output.shape[1:])

    # clean-up
    if is_cuda_array(input):
        gc.collect()
        cp._default_memory_pool.free_all_blocks()

    return output
