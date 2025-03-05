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
    batched : bool, optional
        Toggle leading axis ``(-1)`` for broadcasting. The default is ``False``.
    normalize_coord : bool, optional
        Normalize coordinates between -pi and pi. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    """

    def __init__(
        self,
        ishape: ArrayLike,
        coord: ArrayLike,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        batched: bool = False,
        plan: SimpleNamespace | None = None,
        normalize_coord: bool = True,
    ):
        self.signal_ndim = coord.shape[-1]
        self.fourier_ndim = len(coord.shape[:-1])
        self.coord = coord
        self.oversamp = oversamp
        self.eps = eps
        self.batched = batched

        # get input and output shape
        ishape = ishape[-self.signal_ndim :]
        oshape = coord.shape[: self.fourier_ndim]

        # build plan
        if plan is not None:
            self.plan = plan
        else:
            self.plan = __nufft_init__(coord, ishape, oversamp, eps, normalize_coord)

        # initalize operator
        super().__init__(oshape, ishape)

        # enable broadcasting
        if self.batched:
            self.ishape = [-1] + self.ishape
            self.oshape = [-1] + self.oshape

    def _apply(self, input):
        output = _apply(self.plan, input)
        return output.reshape(*output.shape[:-1], *self.coord.shape[:-1])

    def _adjoint_linop(self):
        if self.batched:
            ishape = self.ishape[-self.signal_ndim :]
        else:
            ishape = self.ishape
        return NUFFTAdjoint(
            ishape, self.coord, self.oversamp, self.eps, self.batched, self.plan
        )

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
    batched : bool, optional
        Toggle leading axis ``(-1)`` for broadcasting. The default is ``False``.
    normalize_coord : bool, optional
        Normalize coordinates between -pi and pi. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    """

    def __init__(
        self,
        oshape: ArrayLike,
        coord: ArrayLike,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        batched: bool = False,
        plan: SimpleNamespace | None = None,
        normalize_coord: bool = True,
    ):
        self.signal_ndim = coord.shape[-1]
        self.fourier_ndim = len(coord.shape[:-1])
        self.coord = coord
        self.oversamp = oversamp
        self.eps = eps
        self.batched = batched

        # get input and output shape
        ishape = coord.shape[: self.fourier_ndim]
        oshape = oshape[-self.signal_ndim :]

        # build plan
        if plan is not None:
            self.plan = plan
        else:
            self.plan = __nufft_init__(coord, oshape, oversamp, eps, normalize_coord)

        # initalize operator
        super().__init__(oshape, ishape)

        # enable broadcasting
        if self.batched:
            self.ishape = [-1] + self.ishape
            self.oshape = [-1] + self.oshape

    def _apply(self, input):
        input = input.reshape(*input.shape[: -self.fourier_ndim], -1)
        return _apply_adj(self.plan, input)

    def _adjoint_linop(self):
        if self.batched:
            oshape = self.oshape[-self.signal_ndim :]
        else:
            oshape = self.oshape
        return NUFFT(
            oshape, self.coord, self.oversamp, self.eps, self.batched, self.plan
        )
