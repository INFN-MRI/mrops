"""Monkey patching Linop."""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import sigpy.linop as original_linop


def _patched_check_shape_positive(shape):
    if not all((s > 0 or s == -1) for s in shape):
        raise ValueError("Shapes must be positive or -1, got {}".format(shape))


# Patch the function in the original module
original_linop._check_shape_positive = _patched_check_shape_positive

# Now import all names from sigpy.linop, which now contains the patched version
from sigpy.linop import *  # noqa

# Explicitly bind the patched _check_shape_positive into our module's namespace
_check_shape_positive = original_linop._check_shape_positive
