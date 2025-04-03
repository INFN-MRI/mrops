"""GROG MRI operator."""

__all__ = ["GrogMR"]

from numpy.typing import NDArray

from .._sigpy import linop
from .._sigpy.linop import Resize

from .. import grog
from ..base import FFT, MultiIndex

from mrinufft._array_compat import with_numpy_cupy


@with_numpy_cupy
def GrogMR(
    ishape: list[int] | tuple[int],
    input: NDArray[complex],
    coords: NDArray[float],
    train_data: NDArray[complex],
    lamda: float = 0.0,
    stack_axes: list[int] | tuple[int] | None = None,
    oversamp: float | list[float] | tuple[float] | None = None,
    radius: float = 0.75,
    precision: int = 1,
    weighting_mode: str = "distance",
    grid: bool = False,
) -> tuple[NDArray[complex], linop.Linop]:
    """
    Single coil GROG MR operator.

    Parameters
    ----------
    ishape : list[int] | tuple[int]
        Input shape ``(ny, nx)`` (2D)
        or ``(nz, ny, nx)`` (3D).
    input : NDArray[complex]
        Input Fourier space data.
    coords : NDArray[int]
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    train_data : NDArray[complex]
        Calibration region data of shape ``(nc, nz, ny, nx)`` or ``(nc, ny, nx)``.
        Usually a small portion from the center of kspace.
    lamda : float, optional
        Tikhonov regularization parameter.  Set to 0 for no
        regularization. Defaults to ``0.01``.
    stack_axes: list[int] | tuple[int], optional
        Index marking stack axes. The default is ``None``,
    oversamp: float | list[float] | tuple[float] | None, optional
        Cartesian grid oversampling factor. If scalar, assume
        same oversampling for all spatial dimensions.
        The default is ``1.0`` (2D MRI) or ``(1.0, 1.0, 1.2)`` (3D MRI).
    radius: float, optional
        Spreading radius. The default is ``0.75``.
    precision: int, optional
        Number of decimal digits in GROG kernel power. The default is ``1``.
    weighting_mode: str, optional
        Non Cartesian samples accumulation mode. Can be:

            * ``"average"``: arithmetic average.
            * ``"distance"``: weight according to distance.

        The default is ``"distance"``.
    grid: bool, optional
        If True, returns data in a Cartesian grid. Otherwise, return sparse data.
        The default is ``False``.

    """
    if len(ishape) != 2 and len(ishape) != 3:
        raise ValueError("shape must be either (ny, nx) or (nz, ny, nx)")

    # train GROG interpolator
    interpolator = grog.train(train_data, lamda)

    # perform GROG interpolation
    output, indexes, shape = grog.interp(
        interpolator,
        input,
        coords,
        ishape,
        stack_axes,
        oversamp,
        radius,
        precision,
        weighting_mode,
    )

    return output, _GrogMR(output, indexes, ishape, coords, shape, stack_axes, grid)


# %% utils
class _GrogMR(linop.Linop):
    def __init__(self, output, indexes, ishape, coords, shape, stack_axes, grid):
        self._batched = False
        ndim = coords.shape[-1]
        if stack_axes is not None:
            try:
                stack_shape = coords.shape[: len(stack_axes)]
            except Exception:
                stack_shape = (coords.shape[stack_axes],)
        else:
            stack_shape = ()

        # enforce tuple
        shape = tuple(shape)
        stack_shape = tuple(stack_shape)
        ishape = tuple(ishape)

        # build operators
        I = MultiIndex(shape, stack_shape, indexes)
        F = FFT(stack_shape + shape, axes=tuple(range(-ndim, 0)))
        R = Resize(stack_shape + shape, stack_shape + ishape)

        # assemble GROG operator
        if grid:
            output = I.H.apply(output)
            G = F * R
            self.stack_indexes = None
            self.spatial_indexes = None
        else:
            G = I * F * R
            self.stack_indexes = indexes[..., :-1].squeeze()
            self.spatial_indexes = indexes[..., -1]

        self.stack_axes = stack_axes
        self._linop = G
        self.repr_str = "GROG"

        super().__init__(G.oshape, G.ishape)

    def _apply(self, input):
        return self._linop._apply(input)

    def _adjoint_linop(self):
        return self._linop.H

    def _normal_linop(self):
        return self._linop.N

    def broadcast(self, n_batches):
        self._batched = True
        for n in range(len(self._linop.linops)):
            self._linop.H.linops[n].ishape = (n_batches,) + tuple(
                self._linop.H.linops[n].ishape
            )
            self._linop.H.linops[n].oshape = (n_batches,) + tuple(
                self._linop.H.linops[n].oshape
            )
        for n in range(len(self._linop.linops)):
            self._linop.linops[n].ishape = (n_batches,) + tuple(
                self._linop.linops[n].ishape
            )
            self._linop.linops[n].oshape = (n_batches,) + tuple(
                self._linop.linops[n].oshape
            )

        self._linop.ishape = (n_batches,) + tuple(self._linop.ishape)
        self._linop.oshape = (n_batches,) + tuple(self._linop.oshape)
        self._linop.H.ishape = (n_batches,) + tuple(self._linop.H.ishape)
        self._linop.H.oshape = (n_batches,) + tuple(self._linop.H.oshape)

        self.ishape = (n_batches,) + tuple(self.ishape)
        self.oshape = (n_batches,) + tuple(self.oshape)

        return self

    def slice(self):
        if self.stack_axes is None:
            return self
        else:
            if self._batched:
                raise ValueError("Batched operators cannot be sliced at the moment")
            stack_indexes = self.stack_indexes
            stack_axes = self.stack_axes

            self.stack_indexes = None
            self.stack_axes = None

            # get number of leading stack axes
            try:
                n_stacks = len(stack_axes)
            except Exception:
                n_stacks = 1

            # remove stack axes
            self._linop.linops[0].H._duplicate_entries = True
            self._linop.linops[0].H.stack_shape = ()
            self._linop.linops[0].stack_shape = ()
            self._linop.linops[0].indexes = self._linop.linops[0].indexes[..., -1]
            self._linop.linops[0].H.indexes = self._linop.linops[0].H.indexes[..., -1]

            # now fix shapes
            for n in range(len(self._linop.linops) - 1):
                self._linop.H.linops[n].ishape = self._linop.H.linops[n].ishape[
                    n_stacks:
                ]
                self._linop.H.linops[n].oshape = self._linop.H.linops[n].oshape[
                    n_stacks:
                ]
            self._linop.H.linops[-1].oshape = self._linop.H.linops[-1].oshape[n_stacks:]
            self._linop.linops[0].ishape = self._linop.linops[0].ishape[n_stacks:]
            for n in range(1, len(self._linop.linops)):
                self._linop.linops[n].ishape = self._linop.linops[n].ishape[n_stacks:]
                self._linop.linops[n].oshape = self._linop.linops[n].oshape[n_stacks:]

            self._linop.ishape = self._linop.ishape[n_stacks:]
            self._linop.H.oshape = self._linop.H.oshape[n_stacks:]
            self.ishape = self.ishape[n_stacks:]

        return self, stack_indexes, stack_axes
