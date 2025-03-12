"""Indexing Linear Operator."""

__all__ = ["MultiIndex", "MultiGrid"]

from numpy.typing import ArrayLike

from .._sigpy.linop import Linop
from .._sigpy import get_device

from ._index import multi_index, multi_grid


class MultiIndex(Linop):
    """
    MultiIndex linear operator.

    Parameters
    ----------
    ishape : ArrayLike[int] | None, optional
        Input shape. Use ``-1`` to enable broadcasting
        across a particular axis (e.g., ``(-1, Ny, Nx)``).
    indexes : ArrayLike
        Fourier domain index array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to index over.

    """

    def __init__(
        self,
        ishape: ArrayLike,
        indexes: ArrayLike,
        output: ArrayLike | None = None,
    ):
        self.signal_ndim = indexes.shape[-1]
        self.fourier_ndim = len(indexes.shape[:-1])
        self.indexes = indexes

        # get input and output shape
        oshape = list(ishape[: -self.signal_ndim]) + list(indexes.shape[:-1])

        # initalize operator
        super().__init__(oshape, ishape)
        if output is not None:
            with get_device(indexes) as device:
                xp = device.xp
                self._output = xp.zeros(ishape, dtype=xp.float32)
        else:
            self._output = output

    def _apply(self, input):
        return multi_index(input, self.indexes)

    def _adjoint_linop(self):
        return MultiGrid(self.ishape, self.indexes, self._output)

    def _normal_linop(self):
        return self.H * self


class MultiGrid(Linop):
    """
    MultiIndex linear operator.

    Parameters
    ----------
    oshape : ArrayLike[int] | None, optional
        Output shape. Use ``-1`` to enable broadcasting
        across a particular axis (e.g., ``(-1, Ny, Nx)``).
    indexes : ArrayLike
        Fourier domain index array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to index over.

    """

    def __init__(
        self,
        oshape: ArrayLike,
        indexes: ArrayLike,
        output: ArrayLike | None = None,
    ):
        self.signal_ndim = indexes.shape[-1]
        self.fourier_ndim = len(indexes.shape[:-1])
        self.indexes = indexes

        # get input and output shape
        ishape = list(oshape[: -self.signal_ndim]) + list(indexes.shape[:-1])

        # initalize operator
        super().__init__(oshape, ishape)
        if output is not None:
            with get_device(indexes) as device:
                xp = device.xp
                self._output = xp.zeros(oshape, dtype=xp.float32)
        else:
            self._output = output

    def _apply(self, input):
        return multi_grid(input, self.indexes, self._output)

    def _adjoint_linop(self):
        return MultiIndex(self.oshape, self.indexes, self._output)
