"""Single To Multicoil Encoding Operator."""

__all__ = ["MulticoilOp"]

import numpy as np

from numpy.typing import ArrayLike

from .._sigpy import linop


def MulticoilOp(encoding: linop.Linop, smaps: ArrayLike):
    """
    Transform single Coil encoding to Multicoil.

    Paramaters
    ----------
    encoding: Linop
        Single-coil forward Encoding operator (e.g., FFT or NUFFT).
    smaps: ArrayLike
        Coil Sensitivity maps.

    Returns
    -------
    Linop
        Multi-coil enabled encoding operator.

    """
    squeeze = linop.Reshape(encoding.ishape, (1,) + tuple(encoding.ishape))
    unsqueeze = linop.Reshape((1,) + tuple(encoding.oshape), encoding.oshape)

    # encoding
    nmaps = smaps.shape[0]
    F = linop.Diag(nmaps * [unsqueeze * encoding * squeeze], iaxis=0, oaxis=0)

    # sensitivity
    shape = smaps.shape[1:]
    S = linop.Vstack(
        [squeeze.H * linop.Multiply(shape, smap) for smap in smaps], axis=0
    )

    return F * S
