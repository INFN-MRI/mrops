"""Monkey patching sigpy.interp."""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sigpy.interp import *  # noqa
    from sigpy import interp

__all__ = interp.__all__
