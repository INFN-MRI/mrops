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
    sobolev_width: int = 32,
    ell: int = 16,
    max_iter: int = 10,
    cg_iter: int = 10,
    cg_tol: float = 0.0,
    alpha0: float = 1.0,
    alpha_min: float = 1e-6,
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
    sobolev_width : int, optional
        Sobolev kernel width, i.e., matrix size of the k-space
        region containing expected coil frequences. The default is ``32``.
    ell : int, optional
        Sobolev norm order for regularization. The default is ``16``.
    max_iter : int, optional
        Number of outer (Gauss-Newton) iterations (default is ``10``).
    cg_iter : int, optional
        Number of inner (Conjugate Gradient) iterations (default is ``10``).
    cg_tol : float, optional
         Tolerance for Conjugate Gradient stopping condition (default is ``0.0``).
    alpha0 : float, optional
        Initial regularization parameter (default is ``1.0``).
    alpha_min : float, optional
        Minimum regularization parameter (default is ``1e-6``).
    q : float, optional
        Decay factor for α per outer iteration (default is ``2/3``).
    show_pbar : bool, optional
        Toggle whether show progress bar (default is ``False``).
    leave_pbar : bool, optional
        Toggle whether to leave progress bar after finished (default is ``True``).
    record_time : bool, optional
        Toggle wheter record runtime (default is ``False``).

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
        nlinv = NlinvOp(device, n_coils, shape, sobolev_width=sobolev_width, ell=ell)
    else:  # Non Cartesian
        if shape is None:
            shape = estimate_shape(coords)
        if get_device(coords).id >= 0:
            coords = coords.get()
        nlinv = NlinvOp(
            device, n_coils, shape, coords, oversamp, eps, sobolev_width, ell
        )

    # Enforce shape as list
    if isinstance(shape, xp.ndarray):
        shape = shape.tolist()
    else:
        shape = list(shape)

    # Initialize guess
    if device.id >= 0:
        with device:
            x0 = xp.zeros((n_coils + 1, *shape), dtype=y.dtype)
    else:
        x0 = xp.zeros((n_coils + 1, *shape), dtype=y.dtype)
    x0[0] = 1.0

    # Calculate xhat0
    xhat0 = nlinv.W.H.apply(x0)

    # Run algorithm
    xhat = IrgnmCG(
        nlinv,
        y,
        xhat0,
        max_iter,
        cg_iter,
        cg_tol,
        alpha0,
        alpha_min,
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
    
    rss = (smaps.conj() * smaps).sum(axis=0) ** 0.5
    rho = rho * rss
    smaps = smaps / rss # like SigPy

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
    kw : int, optional
        Sobolev kernel width, i.e., matrix size of the k-space
        region containing expected coil frequences. The default is ``32``.
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
        kw: int = 32,
        ell: int = 16,
    ):
        self.device = device
        self.n_coils = n_coils
        self.matrix_size = matrix_size

        # Compute the Fourier operator
        self.F = self._get_fourier_op(coords, oversamp, eps)

        # Compute the k-space weighting operator W
        self.W = self._get_weighting_op(kw, ell)

        super().__init__()

    def _compute_forward(self, xhat):
        """
        Compute the forward operator G_n(xhat) for MRI encoding.

        Returns
        -------
        linop.Linop
            Forward model G_n(xhat) as a matrix-free linear operator.

        """
        C = self.W.apply(xhat)[1:]  # Coil sensitivity maps

        # Get single coil Encoding operator (FFT or NUFFT)
        F = self.F

        # Get multicoil Encoding operator: A_n = F * S_n
        return MulticoilOp(F, C)

    def _compute_jacobian(self, xhat):
        """
        Compute the Jacobian operator dG_n(xhat).

        Returns
        -------
        sp.linop.Linop
            SigPy linear operator representing the Jacobian.
        """
        x = self.W.apply(xhat)  # Coil sensitivity maps
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
        return linop.Vstack(DF_n, axis=0) * self.W

    def _get_weighting_op(self, kw, ell):
        """
        Compute the k-space weighting operator W.

        Parameters
        ----------
        kw : int
            Sobolev kernel width.
        ell : int
            Order of the Sobolev norm.

        Returns
        -------
        Linop
            SigPy linear operator representing the k-space weighting.

        """
        eye = linop.Identity((1,) + tuple(self.matrix_size.tolist()))

        # Get weighting
        xp = self.device.xp
        with self.device:
            kgrid = xp.meshgrid(
                *[
                    xp.arange(-n // 2, n // 2, dtype=xp.float32) / n
                    for n in tuple(self.matrix_size.tolist())
                ],
                indexing="ij",
            )
        k_norm = sum(ki**2 for ki in kgrid)
        w = linop.Multiply(
            (self.n_coils,) + tuple(self.matrix_size.tolist()),
            1.0 / (1 + kw * k_norm) ** (ell / 2),
        )
        weight = linop.Diag([eye, w], iaxis=0, oaxis=0)

        # Get cartesian FFT
        spatial_dims = len(self.matrix_size)
        fft_axes = tuple(range(-spatial_dims, 0))  # Last dimensions are spatial
        fourier = FFT((self.n_coils,) + tuple(self.matrix_size.tolist()), axes=fft_axes)
        fourier = linop.Diag([eye, fourier], iaxis=0, oaxis=0)

        # Build low pass filtering
        return fourier.H * weight

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
            spatial_dims = len(self.matrix_size)
            fft_axes = tuple(range(-spatial_dims, 0))  # Last dimensions are spatial
            return FFT(
                (self.n_coils,) + tuple(self.matrix_size.tolist()), axes=fft_axes
            )
        else:
            return NUFFT(
                (self.n_coils,) + tuple(self.matrix_size.tolist()),
                coords,
                oversamp,
                eps,
            )
