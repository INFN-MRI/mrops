"""Monkey patching sigpy.wavelet."""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sigpy.wavelet import *  # noqa
    from sigpy import wavelet

__all__ = wavelet.__all__
