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
    coords : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coords : bool, optional
        Normalize coordinates between -pi and pi. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    """

    def __init__(
        self,
        ishape: ArrayLike,
        coords: ArrayLike,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        plan: SimpleNamespace | None = None,
        normalize_coords: bool = True,
    ):
        self.signal_ndim = coords.shape[-1]
        self.fourier_ndim = len(coords.shape[:-1])
        self.coords = coords
        self.oversamp = oversamp
        self.eps = eps

        # get input and output shape
        oshape = list(ishape[: -self.signal_ndim]) + list(coords.shape[:-1])

        # build plan
        if plan is not None:
            self.plan = plan
        else:
            self.plan = __nufft_init__(
                coords, ishape[-self.signal_ndim :], oversamp, eps, normalize_coords
            )

        # initalize operator
        super().__init__(oshape, ishape)

    def _apply(self, input):
        output = _apply(self.plan, input)
        return output.reshape(*output.shape[:-1], *self.coords.shape[:-1])

    def _adjoint_linop(self):
        return NUFFTAdjoint(
            self.ishape, self.coords, self.oversamp, self.eps, self.plan
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
    coords : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT..
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coords : bool, optional
        Normalize coordinates between -pi and pi. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    """

    def __init__(
        self,
        oshape: ArrayLike,
        coords: ArrayLike,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        plan: SimpleNamespace | None = None,
        normalize_coords: bool = True,
    ):
        self.signal_ndim = coords.shape[-1]
        self.fourier_ndim = len(coords.shape[:-1])
        self.coords = coords
        self.oversamp = oversamp
        self.eps = eps

        # get input and output shape
        ishape = list(oshape[: -self.signal_ndim]) + list(coords.shape[:-1])

        # build plan
        if plan is not None:
            self.plan = plan
        else:
            self.plan = __nufft_init__(
                coords, oshape[-self.signal_ndim :], oversamp, eps, normalize_coords
            )

        # initalize operator
        super().__init__(oshape, ishape)

    def _apply(self, input):
        input = input.reshape(*input.shape[: -self.fourier_ndim], -1)
        return _apply_adj(self.plan, input)

    def _adjoint_linop(self):
        return NUFFT(self.oshape, self.coords, self.oversamp, self.eps, self.plan)
