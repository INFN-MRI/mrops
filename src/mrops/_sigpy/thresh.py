"""Monkey patching sigpy.thresh."""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sigpy.thresh import *  # noqa
    from sigpy import thresh

__all__ = thresh.__all__