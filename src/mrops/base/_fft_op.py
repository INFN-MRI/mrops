"""Fast Fourier Transform Linear Operator."""

__all__ = ["FFT", "IFFT"]

from .._sigpy.linop import Linop, Identity

from ._fftc import fft, ifft


class FFT(Linop):
    """
    FFT linear operator.

    Parameters
    ----------
    shape : list[int] | tuple[int]
        Input shape.
    axes : int | list[int] | tuple[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all axes).
    center : bool, optional
        Toggle center iFFT. The default is ``True``.

    """

    def __init__(
        self,
        shape: list[int] | tuple[int],
        axes: int | list[int] | tuple[int] | None = None,
        center: bool = True,
    ):
        self.axes = axes
        self.center = center
        super().__init__(shape, shape)

    def _apply(self, input):
        return fft(input, axes=self.axes, center=self.center)

    def _adjoint_linop(self):
        return IFFT(self.ishape, axes=self.axes, center=self.center)

    def _normal_linop(self):
        return Identity(self.ishape)


class IFFT(Linop):
    """
    IFFT linear operator.

    Parameters
    ----------
    shape : list[int] | tuple[int]
        Input shape.
    axes : int | list[int] | tuple[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all axes).
    center : bool, optional
        Toggle center iFFT. The default is ``True``.

    """

    def __init__(
        self,
        shape: list[int] | tuple[int],
        axes: int | list[int] | tuple[int] | None = None,
        center: bool = True,
    ):
        self.axes = axes
        self.center = center
        super().__init__(shape, shape)

    def _apply(self, input):
        return ifft(input, axes=self.axes, center=self.center)

    def _adjoint_linop(self):
        return FFT(self.ishape, axes=self.axes, center=self.center)

    def _normal_linop(self):
        return Identity(self.ishape)
