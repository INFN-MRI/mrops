"""Batched Encoding Operator."""

__all__ = ["BatchedOp"]

from .._sigpy import linop


def BatchedOp(encoding: linop.Linop, nbatches: int) -> linop.Linop:
    """
    Transform single volume encoding to Batched.

    Paramaters
    ----------
    encoding: Linop
        Single-volume forward Encoding operator.
    nbatches : int
        Batch size

    Returns
    -------
    Linop
        Batched encoding operator.

    """
    if "GROG" in encoding.repr_str.upper():
        return encoding.broadcast(nbatches)

    # manual broadcasting
    squeeze = linop.Reshape(encoding.ishape, (1,) + tuple(encoding.ishape))
    unsqueeze = linop.Reshape((1,) + tuple(encoding.oshape), encoding.oshape)

    # encoding
    op = linop.Diag(nbatches * [unsqueeze * encoding * squeeze], iaxis=0, oaxis=0)
    op.repr_str = "Batched " + encoding.repr_str

    return op
