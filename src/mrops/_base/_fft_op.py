"""Fast Fourier Transform Operator."""

__all__ = ["FFT", "IFFT"]

from .._functional import fft, ifft

from ._base_op import BaseOperator, IdentityOperator


class FFT(BaseOperator):
    """
    FFT operator.

    Parameters
    ----------
    axes : int | list[int] | tuple[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all axes).
    center : bool, optional
        Toggle center iFFT. The default is ``True``.

    """

    def __init__(
        self,
        axes: int | list[int] | tuple[int] | None = None,
        center: bool = True,
    ):
        super().__init__()
        self.axes = axes
        self.center = center

    def _apply(self, input):
        return fft(input, axes=self.axes, center=self.center)

    def _adjoint_op(self):
        return IFFT(axes=self.axes, center=self.center)

    def _normal_op(self):
        return IdentityOperator()


class IFFT:
    """
    IFFT operator.

    Parameters
    ----------
    axes : int | list[int] | tuple[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all axes).
    center : bool, optional
        Toggle center iFFT. The default is ``True``.

    """

    def __init__(
        self,
        axes: int | list[int] | tuple[int] | None = None,
        center: bool = True,
    ):
        super().__init__()
        self.axes = axes
        self.center = center

    def _apply(self, input):
        return ifft(axes=self.axes, center=self.center)

    def _adjoint_op(self):
        return FFT(axes=self.axes, center=self.center)

    def _normal_op(self):
        return IdentityOperator()
