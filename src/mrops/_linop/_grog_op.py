"""GROG MRI operator."""

__all__ = ["GrogMR"]

from numpy.typing import NDArray

from .._sigpy import linop
from .._sigpy.linop import Resize

from .. import grog
from ..base import FFT, MultiIndex
from ..gadgets import BatchedOp


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
    ndim = coords.shape[-1]

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

    # get stack shape
    if stack_axes is not None:
        try:
            stack_shape = coords.shape[:len(stack_axes)]
        except Exception:
            stack_shape = (coords.shape[stack_axes],)
    else:
        stack_shape = ()
    shape = tuple(shape)
        
    # build operators
    I = MultiIndex(shape, stack_shape, indexes)
    F = FFT(stack_shape + shape, axes=tuple(range(-ndim, 0)))
    R = Resize(stack_shape + shape, stack_shape + ishape)

    # assemble GROG operator
    if grid:
        output = I.H.apply(output)
        G = F * R
        G.stack_indexes = indexes[..., :-1]
        G.spatial_indexes = indexes[..., -1]
    else:
        G = I * F * R
        G.stack_indexes = None
        G.spatial_indexes = None

    # add batch axes
    # batch_axes = input.shape[: -len(indexes.shape[:-1])]
    # for ax in batch_axes:
    #     G = BatchedOp(G, ax)

    # improve representation
    G.stack_axes = stack_axes
    G.repr_str = "GROG Linop"
    
    return output, G
