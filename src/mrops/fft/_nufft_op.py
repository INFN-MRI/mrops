"""Non-Uniform Fast Fourier Transform Linear Operator."""

__all__ = ["NUFFT", "NUFFTAdjoint"]

from types import SimpleNamespace

from numpy.typing import ArrayLike

from .._sigpy.linop import Linop

from ._nufft import __nufft_init__, _apply, _apply_adj

class NUFFT(Linop):
    """
    NUFFT linear operator.

    Parameters
    ----------
    ishape : ArrayLike[int] | None, optional
        Input shape. Use ``-1`` to enable broadcasting
        across a particular axis (e.g., ``(-1, Ny, Nx)``).
    coord : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ndim determines the number of dimensions to apply the nufft.
        ``coord[..., i]`` should be scaled to have its range between
        ``-n_i // 2``, and ``n_i // 2``.
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.

    """

    def __init__(
        self,
        ishape: ArrayLike,
        coord: ArrayLike,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        plan: SimpleNamespace | None = None,
    ):
        self.signal_ndim = coord.shape[-1]
        self.fourier_ndim = len(coord.shape[:-1])
        self.coord = coord
        self.oversamp = oversamp
        self.eps = eps
        
        # get input and output shape
        ishape = ishape[-self.signal_ndim:]
        oshape = coord.shape[:self.fourier_ndim]
        
        # build plan
        if plan is not None:
            self.plan = plan
        else:
            self.plan = __nufft_init__(coord, ishape, oversamp, eps)
            
        # initalize operator
        super().__init__(oshape, ishape)
        
        # enable broadcasting
        self.ishape = [-1] + self.ishape
        self.oshape = [-1] + self.oshape

    def _apply(self, input):
        output = _apply(self.plan, input)
        return output.reshape(*output.shape[:-self.fourier_ndim+1], *self.coord.shape[:-1])

    def _adjoint_linop(self):
        ishape = self.ishape[-self.signal_ndim:]
        return NUFFTAdjoint(ishape, self.coord, self.oversamp, self.eps, self.plan)

    def _normal_linop(self):
        return self.H * self


class NUFFTAdjoint(Linop):
    """
    NUFFT Adjoint linear operator.

    Parameters
    ----------
    oshape : ArrayLike[int] | None, optional
        Output shape. Use ``-1`` to enable broadcasting
        across a particular axis (e.g., ``(-1, Ny, Nx)``).
    coord : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ndim determines the number of dimensions to apply the nufft.
        ``coord[..., i]`` should be scaled to have its range between
        ``-n_i // 2``, and ``n_i // 2``.
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.

    """

    def __init__(
        self,
        oshape: ArrayLike,
        coord: ArrayLike,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        plan: SimpleNamespace | None = None,
    ):
        self.signal_ndim = coord.shape[-1]
        self.fourier_ndim = len(coord.shape[:-1])
        self.coord = coord
        self.oversamp = oversamp
        self.eps = eps
        
        # get input and output shape
        ishape = coord.shape[:self.fourier_ndim]
        oshape = oshape[-self.signal_ndim:]
        
        # build plan
        if plan is not None:
            self.plan = plan
        else:
            self.plan = __nufft_init__(coord, oshape, oversamp, eps)
            
        # initalize operator
        super().__init__(oshape, ishape)
        
        # enable broadcasting
        self.ishape = [-1] + self.ishape
        self.oshape = [-1] + self.oshape

    def _apply(self, input):
        input = input.reshape(*input.shape[:-self.fourier_ndim], -1)
        return _apply_adj(self.plan, input)

    def _adjoint_linop(self):
        oshape = self.oshape[-self.signal_ndim:]
        return NUFFT(oshape, self.coord, self.oversamp, self.eps, self.plan)
