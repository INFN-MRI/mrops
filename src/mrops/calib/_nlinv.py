"""NLINV Operator."""

__all__ = ["nlinv_calib"]

from numpy.typing import ArrayLike

from ..base import NonLinop

from mrinufft._array_compat import with_numpy_cupy
from mrinufft._array_compat import get_array_module

from .._sigpy import get_device
from .._sigpy import linop
from .._sigpy import estimate_shape

from ..base import FFT, NUFFT
from ..gadgets import MulticoilOp
from ..solvers import IrgnmCG


@with_numpy_cupy
def nlinv_calib(
    y: ArrayLike,
    shape: ArrayLike | None = None,
    coords: ArrayLike | None = None,
    oversamp: float = 1.25,
    eps: float = 1e-3,
    ell: int = 16,
    max_iter: int = 10,
    cg_iter: int = 10,
    cg_tol: float = 0.0,
    alpha0: float = 1.0,
    q: float = 2 / 3,
    show_pbar: bool = False,
    leave_pbar: bool = True,
    record_time: bool = False,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Estimate coil sensitivity maps using NLINV.

    Parameters
    ----------
    y : ArrayLike
        Measured k-space data of shape ``(n_coils, ...)``
    shape : ArrayLike[int]
        Image dimensions (e.g., ``(nz, ny, nx)`` for 3D or ``(ny, nx)`` for 2D).
        Used only for Non Cartesian Datasets.
    coords : ArrayLike, optional
        k-space trajectory for NUFFT (``None`` for Cartesian).
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
        Used only for Non Cartesian Datasets.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
        Used only for Non Cartesian Datasets.
    ell : int, optional
        Sobolev norm order for regularization. The default is ``16``.

    Returns
    -------
    smaps : ArrayLike
        Coil sensitivity maps of shape ``(n_coils, *shape)``
    image : ArrayLike
        Reconstructed magnetization of shape ``(*shape)``

    """
    xp = get_array_module(y)
    device = get_device(y)
    n_coils = y.shape[0]

    # Pre-normalize data
    y = y / (y * y.conj()).sum() ** 0.5

    # Determine type of acquisition
    if coords is None:  # Cartesian
        shape = y.shape[1:]
        nlinv = NlinvOp(device, n_coils, shape, ell=ell)
    else:  # Non Cartesian
        if shape is None:
            shape = estimate_shape(coords)
        if get_device(coords).id >= 0:
            coords = coords.get()
        nlinv = NlinvOp(device, n_coils, shape, coords, oversamp, eps, ell)

    # Enforce shape as list
    if isinstance(shape, xp.ndarray):
        shape = shape.tolist()
    else:
        shape = list(shape)

    # Initialize guess
    if device.id >= 0:
        with device:
            xhat0 = xp.zeros((n_coils + 1, *shape), dtype=y.dtype)
    else:
        xhat0 = xp.zeros((n_coils + 1, *shape), dtype=y.dtype)
    xhat0[0] = 1.0

    # Run algorithm
    xhat = IrgnmCG(
        nlinv,
        y,
        xhat0,
        max_iter,
        cg_iter,
        cg_tol,
        alpha0,
        q,
        show_pbar,
        leave_pbar,
        record_time,
    ).run()

    # Invert transformation
    x = nlinv.W.apply(xhat)

    # Post processing
    smaps = x[1:]
    rho = x[0]
    rho = rho * (smaps.conj() * smaps).sum(axis=0) ** 0.5

    return smaps, rho


class NlinvOp(NonLinop):
    """
    Nonlinear operator for calibrationless parallel MRI reconstruction (NLINV).

    This class models the encoding operator A_n = F * S_n, where:
    - F is either an FFT (Cartesian) or NUFFT (Non-Cartesian).
    - S_n is a pointwise multiplication with coil sensitivity maps.

    The Jacobian dA_n is automatically derived.

    Parameters
    ----------
    device : str
        Computational device (``"cpu"`` or ``cuda:n``).
    n_coils : int
        Number of coil channels.
    matrix_size : ArrayLike[int]
        Image dimensions (e.g., ``(nz, ny, nx)`` for 3D or ``(ny, nx)`` for 2D).
    coords : ArrayLike, optional
        k-space trajectory for NUFFT (``None`` for Cartesian).
    oversamp : float, optional
        Oversampling factor. The default is ``1.25``.
    eps : float, optional
        Desired numerical precision. The default is ``1e-6``.
    ell : int, optional
        Sobolev norm order for regularization. The default is ``16``.

    """

    def __init__(
        self,
        device: str,
        n_coils: int,
        matrix_size: ArrayLike,
        coords: ArrayLike | None = None,
        oversamp: float = 1.25,
        eps: float = 1e-3,
        ell: int = 16,
    ):
        self.device = device
        self.n_coils = n_coils
        self.matrix_size = matrix_size

        # Compute the Fourier operator
        self.F = self._get_fourier_op(coords, oversamp, eps)

        # Compute the k-space weighting operator W
        self.weight = self._get_weighting_op(ell)
        self.W = self.weight * self._get_cartesian_fft_op()

        super().__init__()

    def _compute_forward(self, x):
        """
        Compute the forward operator G_n(x) for MRI encoding.

        Returns
        -------
        linop.Linop
            Forward model G_n(x) as a matrix-free linear operator.

        """
        C = self.W.apply(x)[1:]  # Coil sensitivity maps

        # Get single coil Encoding operator (FFT or NUFFT)
        F = self.F

        # Get multicoil Encoding operator: A_n = F * S_n
        G_n = MulticoilOp(F, C)

        return G_n

    def _compute_jacobian(self, x):
        """
        Compute the Jacobian operator dF(x).

        Returns
        -------
        sp.linop.Linop
            SigPy linear operator representing the Jacobian.
        """
        M = x[0]
        C = x[1:]

        # Get single coil encoding operator (FFT or NUFFT)
        F = self.F

        # PF * (M * dC_n + dM * C_n for n in range(self.n_coils+1))
        unsqueeze_ksp = linop.Reshape([1] + F.oshape, F.oshape)
        unsqueeze_im = linop.Reshape(
            (1,) + tuple(self.matrix_size.tolist()), tuple(self.matrix_size.tolist())
        )
        DF_n = []
        for n in range(self.n_coils):
            DF_n.append(
                unsqueeze_ksp
                * F
                * unsqueeze_im.H
                * (
                    unsqueeze_im
                    * linop.Multiply(self.matrix_size.tolist(), M)
                    * linop.Slice(
                        (self.n_coils + 1,) + tuple(self.matrix_size.tolist()), n + 1
                    )
                    + unsqueeze_im
                    * linop.Multiply(self.matrix_size.tolist(), C[n])
                    * linop.Slice(
                        (self.n_coils + 1,) + tuple(self.matrix_size.tolist()), 0
                    )
                )
            )
        DG_n = linop.Vstack(DF_n, axis=0) * self.W

        return DG_n

    def _get_weighting_op(self, ell):
        """
        Compute the k-space weighting operator W.

        Parameters
        ----------
        ell : int
            Order of the Sobolev norm.

        Returns
        -------
        Linop
            SigPy linear operator representing the k-space weighting.

        """
        xp = self.device.xp
        eye = linop.Identity((1,) + tuple(self.matrix_size.tolist()))
        with self.device:
            kgrid = xp.meshgrid(
                *[xp.fft.fftfreq(n) for n in tuple(self.matrix_size.tolist())],
                indexing="ij",
            )
        k_norm = sum(ki**2 for ki in kgrid) ** 0.5
        w = (1 + k_norm) ** (ell / 2)  # l = 16 in the paper
        w = linop.Multiply((self.n_coils,) + tuple(self.matrix_size.tolist()), w)

        return linop.Diag([eye, w], iaxis=0, oaxis=0)

    def _get_cartesian_fft_op(self):
        # Determine number of spatial dimensions (ignoring coil dimension)
        spatial_dims = len(self.matrix_size)
        fft_axes = tuple(range(-spatial_dims, 0))  # Last dimensions are spatial
        fourier = FFT((self.n_coils,) + tuple(self.matrix_size.tolist()), axes=fft_axes)

        # Append identity in front
        eye = linop.Identity((1,) + tuple(self.matrix_size.tolist()))

        return linop.Diag([eye, fourier], iaxis=0, oaxis=0)

    def _get_fourier_op(self, coords, oversamp, eps):
        """
        Return the Fourier transform operator (FFT for Cartesian or NUFFT for non-Cartesian).

        Parameters
        ----------
        coords : ArrayLike
            K-space coordinates. If none, assume Cartesian.
        oversamp : float
            Grid oversampling factor for NUFFT.
        else : float
            Target numerical accuracy for NUFFT.

        Returns
        -------
        Linop
            The appropriate Fourier operator.

        """
        if coords is None:
            return self._get_cartesian_fft_op()
        else:
            return NUFFT(
                (self.n_coils,) + tuple(self.matrix_size.tolist()),
                coords,
                oversamp,
                eps,
            )

    def get_weighting_op(self):
        """
        Return the k-space weighting operator W.

        This can be used externally for pre-processing initial guesses
        or post-processing the final solution.

        Returns
        -------
        sp.linop.Linop
            The k-space weighting operator.
        """
        return self.W
