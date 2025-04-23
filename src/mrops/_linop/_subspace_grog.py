"""Subspace GROG MRI operator."""

__all__ = ["SubspaceGrogMR"]

from numpy.typing import NDArray
import numexpr as ne
import torch

from mrinufft._array_compat import with_torch

from .._sigpy import linop, get_array_module, get_device, resize

from ..base._index import multi_grid, multi_index
from ..base._fftc import fft, ifft
from ..gadgets import BatchedOp

class SubspaceGrogMR(linop.Linop):
    """GROG with subspace projection."""

    def __init__(
        self,
        grogOp: linop.Linop,
        basis: NDArray[complex],
        smaps: NDArray[complex] | None = None,
        toeplitz: bool = False,
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
        if toeplitz:
            normal = _SubspaceGrogMRNormal(grogOp, basis, self._weights, stack_indexes, smaps)
        else:
            normal = None
        self._linop = BatchedOp(grogOp, ncoeff)

        # if smaps is not None:
        #     self._linop = BatchedOp(self._linop, smaps.shape[0])

        self._basis = xp.ascontiguousarray(basis[:, stack_indexes])
        self._stack_indexes = stack_indexes
        self._smaps = smaps

        # get shape
        ishape = list(grogOp.ishape)
        if smaps is not None:
            oshape = [smaps.shape[0], grogOp.oshape[-1]]
        else:
            oshape = [grogOp.oshape[-1]]

        super().__init__(oshape, ishape, repr_str="Subspace Grog")
        self.normal = normal

    def _apply(self, input):
        if self._smaps is not None:
            device = get_device(input)
            xp = device.xp
            with device:
                output = xp.zeros(self.oshape, dtype=input.dtype)
            for n in range(self._smaps.shape[0]):
                _tmp = _mul(self._smaps[n], input)
                _tmp = self._linop(_tmp)
                output[n] = _reduce((_mul(_tmp, self._basis)), -2)
        else:
            _tmp = self._linop(input)
            output[n] = _reduce((_mul(_tmp, self._basis)), -2)
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
            _tmp = _mul(self._weights, input[0, ..., None, :])
            _tmp = _mul(_tmp, self._basis)
            _tmp = self._linop(_tmp)
            output = _mul(self._smaps[0].conj(), _tmp)
            for n in range(1, self._smaps.shape[0]):
                _tmp = _mul(self._weights, input[n, ..., None, :])
                _tmp = _mul(_tmp, self._basis)
                _tmp = self._linop(_tmp)
                output = _sum(output, _mul(self._smaps[n].conj(), _tmp))
                output = output
        else:
            _tmp = _mul(self._weights, input[..., None, :])
            _tmp = _mul(_tmp, self._basis)
            output = self._linop(_tmp)

        return output

    def _adjoint_linop(self):
        return self._fwd
    
    
class _SubspaceGrogMRNormal(linop.Linop):
    """GROG normal operator with subspace projection."""
    
    def __init__(self, grogOp, basis, weights, stack_indexes, smaps):
        device = get_device(basis)
        xp = device.xp
        grid_shape = grogOp._linop.linops[-1].oshape
        center = xp.asarray(grid_shape) // 2
        b = basis[:, stack_indexes]
        
        st_kern = []
        for n in range(basis.shape[0]):
            with device:
                test = xp.zeros([basis.shape[0]] + list(grid_shape), dtype=xp.complex64)
            test[n][*center] = 1.0
            test = fft(test, axes=tuple(range(-len(grid_shape), 0)), norm="ortho")
            test = multi_index(test, grogOp.spatial_indexes, shape=grid_shape)
            test = _reduce(b * test, axis=0)
            test = _mul(test, b.conj())
            test = multi_grid(weights * test, grogOp.spatial_indexes, shape=grid_shape, duplicate_entries=True)
            st_kern.append(test)
            
        # build st kernel
        st_kern = xp.stack(st_kern, axis=0)
        ndims = len(st_kern.shape)-2
        st_kern = xp.fft.fftshift(st_kern, axes=tuple(range(-ndims, 0)))
        st_kern = xp.ascontiguousarray(st_kern)
        self.st_kern = st_kern.reshape(basis.shape[0], basis.shape[0], -1).T.astype(xp.complex64)
        self.grid_shape = list(grid_shape)
        self._smaps = smaps.squeeze()
        
        super().__init__(
            [basis.shape[0]] + grogOp.ishape,
            [basis.shape[0]] + grogOp.ishape,
            repr_str="Subspace Grog Normal",
        )
        
    def _apply(self, input):
        ncoeff = self.oshape[0]
        ndims = len(self.oshape[1:])
        if self._smaps is not None:
            device = get_device(input)
            xp = device.xp
            with device:
                output = xp.zeros(self.oshape, dtype=input.dtype)
            for n in range(self._smaps.shape[0]):
                _tmp = _mul(self._smaps[n], input)
                _tmp = resize(_tmp, [ncoeff] + list(self.grid_shape))
                _tmp = fft(_tmp, axes=tuple(range(-ndims, 0)), center=False, norm="ortho")
                _tmp = _tmp.reshape(ncoeff, -1).T
                _tmp = _batched_matmul(self.st_kern, _tmp)
                _tmp = _tmp.T.reshape(-1, *self.grid_shape)
                _tmp = ifft(_tmp, axes=tuple(range(-ndims, 0)), center=False, norm="ortho")
                _tmp = resize(_tmp, [ncoeff] + list(self.oshape[1:]))
                output = _sum(output, _mul(self._smaps[n].conj(), _tmp))
        else:
            _tmp = resize(input, [ncoeff] + list(self.grid_shape))
            _tmp = fft(_tmp, axes=tuple(range(-ndims, 0)), center=False, norm="ortho")
            _tmp = _tmp.T.resize(-1, ncoeff)
            _tmp = _batched_matmul(self.st_kern, _tmp)
            _tmp = _tmp.resize(*self.grid_shape, -1).T
            _tmp = ifft(_tmp, axes=tuple(range(-ndims, 0)), center=False, norm="ortho")
            output = resize(_tmp, [ncoeff] + list(self.oshape[1:]))
            
        return output

    def _adjoint_linop(self):
        return self
    
    def _normal_linop(self):
        return self
        
# %% utils    
def _mul(a, b):
    if get_device(a).id >= 0:  # GPU case
        return a * b
    else:  # CPU + numexpr
        return (torch.as_tensor(a) * torch.as_tensor(b)).numpy()
    
def _sum(a, b):
    if get_device(a).id >= 0:  # GPU case
        return a + b
    else:  # CPU + numexpr
        return (torch.as_tensor(a) + torch.as_tensor(b)).numpy()

def _reduce(a, axis):
    if get_device(a).id >= 0:  # GPU case
        return a.sum(axis)
    else:  # torch
        return torch.as_tensor(a).sum(dim=axis).numpy()
    
@with_torch
def _batched_matmul(mat, vec):
    return torch.einsum("bij,...bj->...bi", mat, vec)
