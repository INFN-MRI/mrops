"""Monkey patching sigpy.backend."""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sigpy.backend import *  # noqa
    from sigpy import backend

__all__ = backend.__all__
