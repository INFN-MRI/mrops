"""Indexing."""

__all__ = ["multi_index", "multi_grid"]

import numpy as np
from numpy.typing import NDArray

from mrinufft._array_compat import with_numpy_cupy, with_torch

from .._sigpy import get_device


@with_numpy_cupy
def multi_index(
    input: NDArray[complex | float],
    indexes: NDArray[int],
    shape: list[int] | tuple[int],
    stack_shape: list[int] | tuple[int] = None,
) -> NDArray[complex | float]:
    """
    Extract linear indexing over last ``ndim`` dimensions of input.

    Parameters
    ----------
    input : NDArray[complex | float]
        Input ``(..., *stacks, *shape)`` data array to index, with ``...`` being a tuple of
        batching axes and ``shape`` a tuple of Cartesian dimensions.
    indexes : NDArray[int]
        Index array of shape ``(*stacks, nsamples)``.
    shape : list[int] | tuple[int]
        Cartesian matrix shape.
    stack_shape : list[int] | tuple[int], optional
        Stack axes shape. The default is ``None`` (single stack).

    Returns
    -------
    NDArray[complex | float]
        Selected data with shape ``(..., *stacks, nsamples)``.

    """
    # enforce tuple
    shape = tuple(shape)

    # enforce tuple
    shape = tuple(shape)
    if stack_shape is None:
        n_stacks = 0
        stack_shape = ()
    else:
        n_stacks = int(np.prod(stack_shape).item()) if len(stack_shape) != 0 else 0

    if n_stacks > 0:
        reshaped_input = input.reshape(*input.shape[: -len(shape)], n_stacks, -1)
        output = reshaped_input[..., indexes]
        output = output.reshape(*output.shape[:-2], *stack_shape, -1)
    else:
        reshaped_input = input.reshape(*input.shape[: -len(shape)], -1)
        output = reshaped_input[..., indexes]
        output = output.reshape(*output.shape[:-1], -1)

    return output


@with_numpy_cupy
def multi_grid(
    input: NDArray[complex | float],
    indexes: NDArray[int],
    shape: list[int] | tuple[int],
    stack_shape: list[int] | tuple[int] = None,
    duplicate_entries: bool = False,
) -> NDArray[complex | float]:
    """
    Grid values in x to im_size with indices given in idx.

    Parameters
    ----------
    input : NDArray[complex | float]
        Input  sampled data with shape ``(..., *stacks, nsamples)``.
    indexes :  NDArray[int]
        Index array of shape ``(*stacks, nsamples)``.
    shape : list[int] | tuple[int]
        Cartesian matrix shape.
    stack_shape : list[int] | tuple[int], optional
        Stack axes shape. The default is ``None`` (single stack).
    duplicate_entries : bool, optional
        If True, perform accumulation rather than indexing.
        Only meaningful if stack axis is not provided (otherwise assume duplicates
        are already removed).

    Returns
    -------
    NDArray[complex | float]
        Zero-filled Cartesian data with shape ``(..., *stacks, *shape)``.

    Notes
    -----
    Adjoint of multi_index

    """
    device = get_device(input)
    xp = device.xp

    # enforce tuple
    shape = tuple(shape)
    if stack_shape is None:
        n_stacks = 0
        stack_shape = ()
    else:
        n_stacks = int(np.prod(stack_shape).item()) if len(stack_shape) != 0 else 0

    # perform gridding
    with device:
        output = xp.zeros((*input.shape[:-1], *stack_shape, *shape), dtype=input.dtype)
        if n_stacks > 0:
            output = output.reshape(
                *output.shape[: -len(shape) - len(stack_shape)], n_stacks, -1
            )
            output[..., indexes] = input
            output = output.reshape(*input.shape[:-2], *stack_shape, *shape)
        else:
            output = output.reshape(*output.shape[: -len(shape) - len(stack_shape)], -1)
            if duplicate_entries:
                _add_at(output, indexes, input)
            else:
                output[..., indexes] = input
            output = output.reshape(*input.shape[:-1], *shape)

    return output


# %% utils
@with_torch
def _add_at(output, indexes, input):
    output.index_add_(-1, indexes, input)
