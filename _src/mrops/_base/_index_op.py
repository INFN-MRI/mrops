"""Indexing Linear Operator."""

__all__ = ["MultiIndex", "MultiGrid"]

from numpy.typing import NDArray

from .._functional import multi_index, multi_grid

from ._base_op import BaseOperator


class MultiIndex(BaseOperator):
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
        super().__init__()
        self.shape = ishape
        self.stack_shape = stack_shape
        self.indexes = indexes

        # get input and output shape
        oshape = (indexes.shape[0],)
        ishape = stack_shape + ishape

        # save shapes
        self.ishape = ishape
        self.oshape = oshape

    def _apply(self, input):
        return multi_index(input, self.indexes, self.shape, self.stack_shape)

    def _adjoint_op(self):
        output = MultiGrid(self.shape, self.stack_shape, self.indexes)
        output.ishape = self.oshape
        output.oshape = self.ishape
        return output


class MultiGrid(BaseOperator):
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
        super().__init__()
        self.shape = oshape
        self.stack_shape = stack_shape
        self.indexes = indexes
        self._duplicate_entries = False

        # get input and output shape
        ishape = (indexes.shape[0],)
        oshape = stack_shape + oshape

        # save shapes
        self.ishape = ishape
        self.oshape = oshape

    def _apply(self, input):
        return multi_grid(
            input, self.indexes, self.shape, self.stack_shape, self._duplicate_entries
        )

    def _adjoint_op(self):
        output = MultiIndex(self.shape, self.stack_shape.self.indexes)
        output.ishape = self.oshape
        output.oshape = self.ishape
        return output
