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
    shape : ArrayLike[int] | None, optional
        Input shape. Use ``-1`` to enable broadcasting
        across a particular axis (e.g., ``(-1, Ny, Nx)``).
    axes :  ArrayLike[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all axes).
    center : bool, optional
        Toggle center iFFT. The default is ``True``.
    batched : bool, optional
        Toggle leading axis ``(-1)`` for broadcasting. The default is ``False``.

    """

    def __init__(
        self,
        shape: ArrayLike,
        axes: ArrayLike | None = None,
        center: bool = True,
        batched: bool = False,
    ):
        self.axes = axes
        self.center = center
        self.batched = batched

        super().__init__(shape, shape)

        # enable broadcasting
        if self.batched:
            self.ishape = [-1] + self.ishape
            self.oshape = [-1] + self.oshape

    def _apply(self, input):
        return fft(input, axes=self.axes, center=self.center)

    def _adjoint_linop(self):
        if self.batched:
            ishape = self.ishape[1:]
        else:
            ishape = self.ishape
        return IFFT(ishape, axes=self.axes, center=self.center, batched=self.batched)

    def _normal_linop(self):
        return Identity(self.ishape)


class IFFT(Linop):
    """
    IFFT linear operator.

    Parameters
    ----------
    shape : ArrayLike[int] | None, optional
        Input shape. Use ``-1`` to enable broadcasting
        across a particular axis (e.g., ``(-1, Ny, Nx)``).
    axes :  ArrayLike[int] | None, optional
        Axes over which to compute the FFT.
        The default is ``None`` (all axes).
    center : bool, optional
        Toggle center iFFT. The default is ``True``.
    batched : bool, optional
        Toggle leading axis ``(-1)`` for broadcasting. The default is ``False``.

    """

    def __init__(
        self,
        shape: ArrayLike,
        axes: ArrayLike | None = None,
        center: bool = True,
        batched: bool = False,
    ):
        self.axes = axes
        self.center = center
        self.batched = batched

        super().__init__(shape, shape)

        # enable broadcasting
        if self.batched:
            self.ishape = [-1] + self.ishape
            self.oshape = [-1] + self.oshape

    def _apply(self, input):
        return ifft(input, axes=self.axes, center=self.center)

    def _adjoint_linop(self):
        if self.batched:
            ishape = self.ishape[1:]
        else:
            ishape = self.ishape
        return FFT(ishape, axes=self.axes, center=self.center, batched=self.batched)

    def _normal_linop(self):
        return Identity(self.ishape)
