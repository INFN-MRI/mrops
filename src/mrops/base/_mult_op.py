"""Element-wise multiplication."""

__all__ = ["Multiply"]

from numpy.typing import ArrayLike

from sigpy.linop import _get_multiply_adjoint_sum_axes

from .._sigpy import linop


class Multiply(linop.Multiply):
    """
    Multiplication linear operator.

    Parameters
    ----------
    ishape : ArrayLike[int]
        Input shape. Use ``-1`` to enable broadcasting
        across a particular axis (e.g., ``(-1, Ny, Nx)``).
    mult : ArrayLike
        Array to multiply.
    batched : bool, optional
        Toggle leading axis ``(-1)`` for broadcasting. The default is ``False``.

    """

    def __init__(
        self,
        ishape: ArrayLike,
        mult: ArrayLike,
        conj: bool = False,
        batched: bool = False,
    ):
        super().__init__(ishape, mult)
        self.batched = batched
        if self.batched:
            self.ishape = [-1] + list(self.ishape)
            self.oshape = [-1] + list(self.oshape)

    def _adjoint_linop(self):
        return _AdjointMultiply(self)


class _AdjointMultiply(linop.Linop):
    def __init__(self, multOp: Multiply):
        if multOp.batched:
            ishape = multOp.ishape[1:]
            oshape = multOp.oshape[1:]
        else:
            ishape = multOp.ishape
            oshape = multOp.oshape

        # Build adjoint
        sum_axes = _get_multiply_adjoint_sum_axes(oshape, ishape, multOp.mshape)
        M = Multiply(oshape, multOp.mult, not multOp.conj)
        S = linop.Sum(M.oshape, sum_axes)
        R = linop.Reshape(ishape, S.oshape)
        self._linops = R * S * M
        super().__init__(self._linops.oshape, self._linops.ishape)

        # Handle batched case
        self.batched = multOp.batched
        if self.batched:
            self.ishape = [-1] + list(self.ishape)
            self.oshape = [-1] + list(self.oshape)
            for _linop in self._linops.linops:
                _linop.batched = True
                _linop.ishape = [-1] + list(_linop.ishape)
                _linop.oshape = [-1] + list(_linop.oshape)

    def _apply(self, input):
        return self._linops._apply(input)

    def _adjoint_linop(self):
        if self.batched:
            ishape = self.ishape[1:]
        else:
            ishape = self.ishape
        return Multiply(
            ishape,
            self._linops.linops[-1].mult,
            not self._linops.linops[-1].conj,
            self.batched,
        )
