"""Indexing."""

__all__ = ["multi_index", "multi_grid"]

from numpy.typing import ArrayLike

from mrinufft._array_compat import with_numpy_cupy

from .._sigpy import get_device


@with_numpy_cupy
def multi_index(input: ArrayLike, indexes: ArrayLike) -> ArrayLike:
    """
    Extract linear indexing over last ``ndim`` dimensions of input.

    Parameters
    ----------
    input : ArrayLike
        Input ``(*B, *shape)`` data array to index, with ``B`` being a tuple of
        batching axes and ``shape`` a tuple of Cartesian dimensions.
    indexes : ArrayLike
        Index array of shape ``(*I, ndims)`` with ``I`` being a tuple of
        sampling dimensions.

    Returns
    -------
    ArrayLike
        Selected data with shape ``(*B, *I)``

    """
    ndims = indexes.shape[-1]
    tup = (slice(None),) * (input.ndim - ndims) + tuple(indexes.moveaxis(-1, 0))
    return input[tup]


@with_numpy_cupy
def multi_grid(
    input: ArrayLike,
    indexes: ArrayLike,
    shape: ArrayLike,
    output: ArrayLike | None = None,
) -> ArrayLike:
    """
    Grid values in x to im_size with indices given in idx.

    Parameters
    ----------
    input : ArrayLike
        Input  sampled data with shape ``(*B, *I)``, with ``B`` being a tuple of
        batching axes and ``I`` a tuple of sampling dimensions.
    indexes : ArrayLike
        Index array of shape ``(*I, ndims)`` with ``I`` being a tuple of
        sampling dimensions.
    shape : ArrayLike[int]
        Cartesian shape.
    output : ArrayLike | None, optional
        Output ``(*B, *shape)`` data array to index, with ``B`` being a tuple of
        batching axes and ``shape`` a tuple of Cartesian dimensions. If ``None``,
        is initialized to zeros.

    Returns
    -------
    ArrayLike
        Zero-filled Cartesian data with shape ``(*B, *shape)``

    Notes
    -----
    Adjoint of multi_index

    """
    indexes = _ravel(indexes, shape, dim=-1)

    # reshape to (..., -1)
    ndim = len(indexes.shape)
    input = input.reshape(*input.shape[:-ndim], -1)
    indexes = indexes.ravel()

    # perform filling
    batch_shape = input.shape[:-1]
    with get_device(input) as device:
        if output is None:
            output = device.xp.zeros((*batch_shape, *shape), dtype=input.dtype)
        else:
            output[:] = 0.0
        output = output.reshape((*batch_shape, -1))
        output = output.index_add_(-1, indexes, input)
        output = output.reshape(*batch_shape, *shape)

    return output


# %% subroutines
def _ravel(indexes, shape):
    ndim = indexes.shape[-1]
    with get_device(indexes) as device:
        unfolding = [1] + shape[: ndim - 1]
        flattened_indexes = device.xp.asarray(unfolding, dtype=indexes.dtype) * indexes
        return flattened_indexes.sum(axis=-1)
