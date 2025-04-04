"""Subspace GROG MRI operator."""

__all__ = ["SubspaceGrogMR"]

from numpy.typing import NDArray

from .._sigpy import linop, get_array_module, get_device

from ..gadgets import BatchedOp


class SubspaceGrogMR(linop.Linop):
    """GROG with subspace projection."""

    def __init__(
        self,
        grogOp: linop.Linop,
        basis: NDArray[complex],
        smaps: NDArray[complex] | None = None,
    ):
        # slice operator
        grogOp, stack_indexes = grogOp.slice()
        ncoeff = basis.shape[0]

        # get weights
        # count elements
        xp = get_array_module(grogOp.spatial_indexes)
        _, idx, counts = xp.unique(
            grogOp.spatial_indexes, return_inverse=True, return_counts=True
        )
        weights = counts[idx]

        # count
        weights = weights.astype(xp.float32)
        self._weights = 1.0 / weights

        if smaps is not None:
            smaps = xp.expand_dims(smaps, -len(grogOp.ishape) - 1)

        # assign operator
        self._smaps = smaps
        self._linop = BatchedOp(grogOp, ncoeff)

        # if smaps is not None:
        #     self._linop = BatchedOp(self._linop, smaps.shape[0])

        self._basis = basis
        self._stack_indexes = stack_indexes
        self._smaps = smaps

        # get shape
        ishape = list(grogOp.ishape)
        if smaps is not None:
            oshape = [smaps.shape[0], grogOp.oshape[-1]]
        else:
            oshape = [grogOp.oshape[-1]]

        super().__init__(oshape, ishape, repr_str="Subspace Grog")

    def _apply(self, input):
        if self._smaps is not None:
            device = get_device(input)
            xp = device.xp
            with device:
                output = xp.zeros(self.oshape, dtype=input.dtype)
            for n in range(self._smaps.shape[0]):
                _tmp = self._smaps[n] * input
                _tmp = self._linop(_tmp)
                output[n] = (self._basis[:, self._stack_indexes] * _tmp).sum(axis=-2)
        else:
            _tmp = self._linop(input)
            output[n] = (self._basis[:, self._stack_indexes] * _tmp).sum(axis=-2)
        return output

    def _adjoint_linop(self):
        return _SubspaceGrogMRAdjoint(self)

    def broadcast(self, n_batches):  # noqa
        self._linop = self._linop.broadcast(n_batches)

        self.ishape = [n_batches] + list(self.ishape)
        self.oshape = [n_batches] + list(self.oshape)

        if self.adj is not None:
            self.H.ishape = [n_batches] + list(self.H.ishape)
            self.H.oshape = [n_batches] + list(self.H.oshape)

        return self


class _SubspaceGrogMRAdjoint(linop.Linop):
    """GROG adjoint with subspace projection."""

    def __init__(self, subspaceGrogOp):
        self._fwd = subspaceGrogOp
        self._linop = subspaceGrogOp._linop.H
        self._basis = subspaceGrogOp._basis.conj()
        self._stack_indexes = subspaceGrogOp._stack_indexes
        self._smaps = subspaceGrogOp._smaps
        self._weights = subspaceGrogOp._weights

        super().__init__(
            subspaceGrogOp.ishape,
            subspaceGrogOp.oshape,
            repr_str="Subspace Grog Adjoint",
        )

    def _apply(self, input):
        if self._smaps is not None:
            _tmp = self._weights * input[0, ..., None, :]
            _tmp = self._basis[:, self._stack_indexes] * _tmp
            _tmp = self._linop(_tmp)
            output = self._smaps[0].conj() * _tmp
            for n in range(1, self._smaps.shape[0]):
                _tmp = self._weights * input[n, ..., None, :]
                _tmp = self._basis[:, self._stack_indexes] * _tmp
                _tmp = self._linop(_tmp)
                output += self._smaps[n].conj() * _tmp
        else:
            _tmp = self._weights * input[..., None, :]
            _tmp = self._basis[:, self._stack_indexes] * _tmp
            output = self._linop(_tmp)

        return output

    def _adjoint_linop(self):
        return self._fwd
