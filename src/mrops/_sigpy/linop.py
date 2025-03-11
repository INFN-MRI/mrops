"""Monkey patching Linop."""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import sigpy.linop as original_linop
    from sigpy import get_device


import numpy as np


# Patch the _check_shape_positive function
def _patched_check_shape_positive(shape):
    if not all((s > 0 or s == -1) for s in shape):
        raise ValueError("Shapes must be positive or -1, got {}".format(shape))


original_linop._check_shape_positive = _patched_check_shape_positive


# Patch the Embed class
class PatchedEmbed(original_linop.Linop):  # noqa
    """Embed input into a zero array with the given shape and index.

    Given input `input` and index `idx`,
    returns output with `output[idx] = input`.

    Args:
        oshape (tuple of ints): output shape.
        idx (slice or tuple of slices): Index.
    """

    def __init__(self, oshape, idx):
        self.idx = idx
        ishape = np.empty(oshape)[idx].shape
        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = get_device(input)
        with device:
            output = device.xp.zeros(self.oshape, dtype=input.dtype)
            output[self.idx] = input
        return output

    def _adjoint_linop(self):
        return original_linop.Slice(self.oshape, self.idx)


# Replace the original Embed class with the patched version
original_linop.Embed = PatchedEmbed

# Import everything from sigpy.linop, now with the patched Embed and _check_shape_positive
from sigpy.linop import *  # noqa

# Explicitly bind the patched elements into our module's namespace
_check_shape_positive = original_linop._check_shape_positive
Embed = original_linop.Embed
