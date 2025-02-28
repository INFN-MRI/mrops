"""Monkey patching sigpy.prox."""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sigpy.prox import *  # noqa
