"""Fast Fourier Transform Linear Operator."""

__all__ = ["FFT", "IFFT"]

from numpy.typing import ArrayLike

from .._sigpy.linop import Linop, Identity

from ._fftc import fft, ifft


class FFT(Linop):
    """
    FFT linear operator.

    Parameters
    ----------
    shape : ArrayLike[int]
        Input shape. Use ``-1`` to enable broadcasting
        across a particular axis (e.g., ``(-1, Ny, Nx)``).
    axes :  ArrayLike[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all axes).
    center : bool, optional
        Toggle center iFFT. The default is ``True``.

    """

    def __init__(
        self,
        shape: ArrayLike,
        axes: ArrayLike | None = None,
        center: bool = True,
    ):
        self.axes = axes
        self.center = center
        super().__init__(shape, shape)

    def _apply(self, input):
        if self.axes is None and self.batched:
            axes = list(range(-len(self.ishape, 0)))
        else:
            axes = self.axes
        return fft(input, axes=axes, center=self.center)

    def _adjoint_linop(self):
        return IFFT(self.ishape, axes=self.axes, center=self.center)

    def _normal_linop(self):
        return Identity(self.ishape)


class IFFT(Linop):
    """
    IFFT linear operator.

    Parameters
    ----------
    shape : ArrayLike[int]
        Input shape. Use ``-1`` to enable broadcasting
        across a particular axis (e.g., ``(-1, Ny, Nx)``).
    axes :  ArrayLike[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all axes).
    center : bool, optional
        Toggle center iFFT. The default is ``True``.

    """

    def __init__(
        self,
        shape: ArrayLike,
        axes: ArrayLike | None = None,
        center: bool = True,
    ):
        self.axes = axes
        self.center = center
        super().__init__(shape, shape)

    def _apply(self, input):
        if self.axes is None and self.batched:
            axes = list(range(-len(self.ishape, 0)))
        else:
            axes = self.axes
        return ifft(input, axes=axes, center=self.center)

    def _adjoint_linop(self):
        return FFT(self.ishape, axes=self.axes, center=self.center)

    def _normal_linop(self):
        return Identity(self.ishape)
