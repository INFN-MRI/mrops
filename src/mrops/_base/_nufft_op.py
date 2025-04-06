"""Non-Uniform Fast Fourier Transform Linear Operator."""

__all__ = ["NUFFT", "NUFFTAdjoint"]

from types import SimpleNamespace

from numpy.typing import NDArray

from .._functional._nufft import __nufft_init__, _apply, _apply_adj

from ._base_op import BaseOperator


class NUFFT(BaseOperator):
    """
    NUFFT operator.

    Parameters
    ----------
    ishape : list[int] | tuple[int]
        Input shape.
    coords : NDArray[float]
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
        ishape: list[int] | tuple[int],
        coords: NDArray[float],
        oversamp: float = 1.25,
        eps: float = 1e-3,
        plan: SimpleNamespace | None = None,
        normalize_coords: bool = True,
    ):
        super().__init__()
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

        # save shapes
        self.ishape = ishape
        self.oshape = oshape

    def _apply(self, input):
        output = _apply(self.plan, input)
        return output.reshape(*output.shape[:-1], *self.coords.shape[:-1])

    def _adjoint_op(self):
        return NUFFTAdjoint(
            self.ishape, self.coords, self.oversamp, self.eps, self.plan
        )


class NUFFTAdjoint(BaseOperator):
    """
    NUFFT Adjoint operator.

    Parameters
    ----------
    oshape : list[int] | tuple[int]
        Output shape.
    coords : NDArray[float]
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
        oshape: list[int] | tuple[int],
        coords: NDArray[float],
        oversamp: float = 1.25,
        eps: float = 1e-3,
        plan: SimpleNamespace | None = None,
        normalize_coords: bool = True,
    ):
        super().__init__()
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

        # save shapes
        self.ishape = ishape
        self.oshape = oshape

    def _apply(self, input):
        input = input.reshape(*input.shape[: -self.fourier_ndim], -1)
        return _apply_adj(self.plan, input)

    def _adjoint_op(self):
        return NUFFT(self.oshape, self.coords, self.oversamp, self.eps, self.plan)