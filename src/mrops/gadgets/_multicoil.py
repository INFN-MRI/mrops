"""Single To Multicoil Encoding Operator."""

__all__ = ["MulticoilOp"]

from numpy.typing import ArrayLike

from .._sigpy import linop

from ._batched import BatchedOp


def MulticoilOp(encoding: linop.Linop, smaps: ArrayLike) -> linop.Linop:
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
    unsqueeze = linop.Reshape((1,) + tuple(encoding.ishape), encoding.ishape)
    F = BatchedOp(encoding, smaps.shape[0])

    # sensitivity
    shape = encoding.ishape
    S = linop.Vstack(
        [unsqueeze * linop.Multiply(shape, smap) for smap in smaps], axis=0
    )

    if encoding.__class__.__name__ == "ToeplitzOp":
        op = S.H * F * S
    else:
        op = F * S
    op.repr_str = "Multicoil " + encoding.repr_str
    return op
