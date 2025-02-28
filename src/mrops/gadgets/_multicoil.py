"""Single To Multicoil Encoding Operator."""

__all__ = ["MulticoilOp"]

from numpy.typing import ArrayLike

from .._sigpy import linop


def MulticoilOp(encoding: linop.Linop, smaps: ArrayLike, batched: bool = False):
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
    shape = smaps.shape[1:]
    return linop.Vstack(
        [encoding * linop.Multiply(shape, smap) for smap in smaps], axis=0
    )
