"""Monkey patching sigpy.config."""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sigpy.config import *  # noqa
