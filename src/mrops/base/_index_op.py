"""Indexing Linear Operator."""

__all__ = ["MultiIndex", "MultiGrid"]

from numpy.typing import NDArray

from .._sigpy.linop import Linop

from ._index import multi_index, multi_grid


class MultiIndex(Linop):
    """
    MultiIndex linear operator.

    Parameters
    ----------
    ishape : list[int] | tuple[int]
        Input shape.
    stack_shape : list[int] | tuple[int]
        Stack shape.
    indexes : NDArray[int]
        Index array of shape ``(*stacks, prod(shape))``.

    """

    def __init__(
        self,
        ishape: list[int] | tuple[int],
        stack_shape: list[int] | tuple[int],
        indexes: NDArray[complex | float],
    ):
        self.shape = ishape
        self.stack_shape = stack_shape
        self.indexes = indexes

        # get input and output shape
        oshape = (indexes.shape[0],)
        ishape = stack_shape + ishape

        # initalize operator
        super().__init__(oshape, ishape)

    def _apply(self, input):
        return multi_index(input, self.indexes, self.shape, self.stack_shape)

    def _adjoint_linop(self):
        output = MultiGrid(self.shape, self.stack_shape, self.indexes)
        output.ishape = self.oshape
        output.oshape = self.ishape
        return output

    def _normal_linop(self):
        return self.H * self


class MultiGrid(Linop):
    """
    MultiIndex linear operator.

    Parameters
    ----------
    oshape : list[int] | tuple[int]
        Output shape.
    stack_shape : list[int] | tuple[int]
        Stack shape.
    indexes : NDArray[int]
        Index array of shape ``(*stacks, prod(shape))``.

    """

    def __init__(
        self,
        oshape: list[int] | tuple[int],
        stack_shape: list[int] | tuple[int],
        indexes: NDArray[complex | float],
    ):
        self.shape = oshape
        self.stack_shape = stack_shape
        self.indexes = indexes
        self._duplicate_entries = False

        # get input and output shape
        ishape = (indexes.shape[0],)
        oshape = stack_shape + oshape

        # initalize operator
        super().__init__(oshape, ishape)

    def _apply(self, input):
        return multi_grid(
            input, self.indexes, self.shape, self.stack_shape, self._duplicate_entries
        )

    def _adjoint_linop(self):
        output = MultiIndex(self.shape, self.stack_shape.self.indexes)
        output.ishape = self.oshape
        output.oshape = self.ishape
        return output
