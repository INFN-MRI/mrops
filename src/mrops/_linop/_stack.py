"""Stack Linear Operators."""

__all__ = ["stack"]

from .._sigpy import linop


def stack(*operators: list[linop.Linop], axis=0) -> linop.Linop:
    """
    Stack linear operators along axis.

    Parameters
    ----------
    *operators : list[linop.Linop]
        List of Linops to be stacked.
    axis : int, optional
        Stacking axis. The default is 0.

    Returns
    -------
    Linop
        Stacked operator.

    """
    nstacks = len(operators)
    shape = operators[0].shape
    _unsqueeze = linop.Reshape((1,) + tuple(shape), tuple(shape))
    _linops = [
        _unsqueeze * operators[n] * linop.Slice(shape, n) for n in range(nstacks)
    ]
    return linop.Vstack(_linops, axis=axis)
