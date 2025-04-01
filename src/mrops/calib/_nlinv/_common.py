"""Base NLINV operator class."""

__all__ = ["BaseNlinvOp"]

from mrinufft._array_compat import get_array_module

from ..._sigpy import linop, Device

from ...base import NonLinop
from ..gadgets import MulticoilOp

from ._nlinv_reg import SobolevOp


class BaseNlinvOp(NonLinop):
    """
    Base NLINV nonlinear operator.

    Holds shared codebase for both Cartesian and Non Cartesian variants.

    Attributes
    ----------
    device: str | int | Device
        Computational device.
    n_coils: int
        Number of coils.
    matrix_size: list[int] | tuple[int]
        Image matrix size ``(nz, ny, nx)`` or ``(ny, nx)``
    kw: float, optional
        Sobolev filter width. The default is ``220.0``.
    ell: int, optional
        Sobolev filter order. The default is ``32``.

    """

    def __init__(
        self,
        device: str | int | Device,
        n_coils: int,
        matrix_size: list[int] | tuple[int],
        kw: float = 220.0,
        ell: int = 32,
    ):
        self.device = device
        self.n_coils = n_coils
        self.matrix_size = matrix_size

        # Compute the k-space weighting operator W
        self._W = self._get_weighting_op(kw, ell)

        super().__init__()

    def W(self, input):
        """Apply Sobolev regularization."""
        output = input.copy()
        for s in range(1, input.shape[0]):
            output[s] = self._W.apply(input[s])

        return output

    def _get_weighting_op(self, kw, ell):
        """Create Sobolev regularization operator."""
        return SobolevOp(self.matrix_size, kw, ell, self.device)

    def _compute_forward(self, xhat):
        """Create forward model operator."""
        x = self.W(xhat)
        smaps = x[1:]
        return MulticoilOp(self._PF, smaps)

    def _compute_jacobian(self, xhat):
        """Compute derivative of forward operator."""
        return _NlinvJacobian(self.matrix_size, self._W, self._PF, self.W(xhat))


class _NlinvJacobian(linop.Linop):
    """Jacobian of NLINV operator."""

    def __init__(self, matrix_size, W, PF, x):
        """Compute derivative of forward operator."""
        try:
            shape = tuple(matrix_size.tolist())
        except Exception:
            shape = tuple(matrix_size)

        # Split input
        rho = x[0]
        smaps = x[1:]
        n_coils = smaps.shape[0]

        # Compute current derivative operator
        # PF * (M * dC_n + dM * C_n for n in range(self.n_coils+1))
        unsqueeze = linop.Reshape([1] + PF.oshape, PF.oshape)
        DF_n = []
        for n in range(n_coils):
            DF_n.append(
                unsqueeze
                * PF
                * (
                    linop.Multiply(shape, rho)
                    * W
                    * linop.Slice((n_coils + 1,) + tuple(shape), n + 1)
                    + linop.Multiply(shape, smaps[n])
                    * linop.Slice((n_coils + 1,) + tuple(shape), 0)
                )
            )

        _linop = linop.Vstack(DF_n, axis=0)

        super().__init__(_linop.oshape, _linop.ishape)
        self._linop = _linop
        self._normal = _NlinvNormal(_linop.ishape, W, PF, x)

    def _apply(self, input):
        return self._linop._apply(input)

    def _normal_linop(self):
        return self._normal

    def _adjoint_linop(self):
        return self._linop.H


class _NlinvNormal(linop.Linop):
    """Normal operator corresponding to NLINV forward pass."""

    def __init__(self, shape, W, PF, x):
        rho = x[0]
        smaps = x[1:]

        # build
        self._W = W
        self.FHF = PF.N
        self.rho = rho
        self.smaps = smaps
        super().__init__(shape, shape)

    def _apply(self, dxhat):
        xp = get_array_module(dxhat)
        dx = self.W(dxhat)

        # Split
        drho_in = dx[0]
        dsmaps_in = dx[1:]

        # Pre-process Fourier Normal operator input
        _tmp = dsmaps_in * self.rho + self.smaps * drho_in

        # Apply Fourier Normal operator
        _tmp = xp.stack([self.FHF.apply(_el) for _el in _tmp])

        # Post-process Fourier Normal operator output
        drho_out = (self.smaps.conj() * _tmp).sum(axis=0)[None, ...]
        dsmaps_out = self.rho.conj() * _tmp

        return self.Wadjoint(xp.concatenate((drho_out, dsmaps_out), axis=0))

    def W(self, input):
        """Apply Sobolev regularization."""
        output = input.copy()
        for s in range(1, input.shape[0]):
            output[s] = self._W.apply(input[s])

        return output

    def Wadjoint(self, input):
        """Apply adjoint of Sobolev regularization."""
        output = input.copy()
        for s in range(1, input.shape[0]):
            output[s] = self._W.H.apply(input[s])

        return output
