"""Monkey patching sigpy.app."""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sigpy.app import *  # noqa
