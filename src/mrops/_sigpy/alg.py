"""Monkey patching sigpy.alg."""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sigpy.alg import *  # noqa
