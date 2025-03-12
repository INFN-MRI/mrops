"""Non Cartesian MRI operator."""

__all__ = ["NonCartesianMR"]

from numpy.typing import ArrayLike

from .._sigpy import linop
from .._sigpy.linop import Multiply

from ..base import NUFFT

from ..toep import ToeplitzOp


class NonCartesianMR(linop.Linop):
    """
    Single coil Non Cartesian MR operator.

    Parameters
    ----------
    ishape : ArrayLike[int] | None, optional
        Input shape ``(ny, nx)`` (2D) or ``(nz, ny, nx)`` (3D).
    coords : ArrayLike
        Fourier domain coordinate array of shape ``(..., ndim)``.
        ``ndim`` determines the number of dimensions to apply the NUFFT.
    weights : ArrayLike | None, optional
        Fourier domain density compensation array for NUFFT (``None`` for Cartesian).
        If not provided, does not perform density compensation. If provided,
        must be shaped ``coords.shape[:-1]``.
    toeplitz : bool | None, optional
        Use Toeplitz PSF to evaluate normal operator.
        The default is ``True`` for 2D imaging and ``False`` for 3D.
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    normalize_coords : bool, optional
        Normalize coordinates between ``-pi`` and ``pi``. If ``False``,
        assume they are correctly normalized already. The default
        is ``True``.

    """

    def __init__(
        self,
        ishape: ArrayLike,
        coords: ArrayLike,
        weights: ArrayLike | None = None,
        toeplitz: bool | None = None,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        normalize_coords: bool = True,
    ):
        if len(ishape) != 2 and len(ishape) != 3:
            raise ValueError("shape must be either (ny, nx) or (nz, ny, nx)")

        # Generate NUFFT operator
        F = NUFFT(ishape, coords, oversamp, eps, normalize_coords=normalize_coords)

        # Density compensation
        if weights is not None:
            PF = Multiply(F.oshape, weights**0.5) * F
        else:
            PF = F

        super().__init__(PF.oshape, PF.ishape)
        self._linop = PF
        self._shape = ishape
        self._coords = coords
        self._weights = weights
        self._oversamp = oversamp
        self._eps = eps
        self._normalize_coords = normalize_coords

        if toeplitz is None:
            if coords.shape[-1] == 2:
                toeplitz = True
            else:
                toeplitz = False
        self._toeplitz = toeplitz

    def _apply(self, input):
        return self._linop._apply(input)

    def _adjoint_linop(self):
        return self._linop.H

    def _normal_linop(self):
        if self._toeplitz is False:
            return self._linop.H * self._linop

        return ToeplitzOp(
            self._shape,
            self._coords,
            self._weights,
            self._oversamp,
            self._eps,
            self._normalize_coords,
        )
