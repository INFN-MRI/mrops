"""Trajectory container."""

__all__ = ["Trajectory"]

from dataclasses import dataclass
from types import ModuleType
from numpy.typing import NDArray, DTypeLike

import numpy as np
import torch

from ..backend import get_device, to_array_module, CUPY_AVAILABLE, Device, NP2TORCH

DeviceType = int | torch.device | Device
if CUPY_AVAILABLE:
    import cupy as cp

    DeviceType = DeviceType | cp.cuda.Device


@dataclass
class Trajectory:
    """
    K-space trajectory representation supporting optional stack dimensions.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (typically 2 or 3).
    nx : int
        Size along the x-axis in k-space.
    ny : int
        Size along the y-axis in k-space.
    kx : NDArray
        Array of x-coordinates in k-space.
    ky : NDArray
        Array of y-coordinates in k-space.
    nz : int, optional
        Size along the z-axis in k-space.
    kz : NDArray, optional
        Array of z-coordinates in k-space.
    slice_axis : NDArray, optional
        Axis for slice indexing.
    contrast_axis : NDArray, optional
        Axis for contrast indexing.
    time_axis : NDArray, optional
        Axis for time/frame indexing.
    nslices : int, optional
        Number of slices.
    ncontrasts : int, optional
        Number of contrasts.
    nframes : int, optional
        Number of frames.

    """

    ndim: int

    # k-space size
    nx: int
    ny: int

    # k-space coordinates
    kx: NDArray
    ky: NDArray

    # Optional fourier axis
    nz: int | None = None
    kz: NDArray | None = None

    # stack axes coordinates
    slice_axis: NDArray | None = None
    contrast_axis: NDArray | None = None
    time_axis: NDArray | None = None

    # stack space size
    nslices: int | None = None
    ncontrasts: int | None = None
    nframes: int | None = None

    def __post_init__(self):  # noqa
        self._hybrid_trajectory = False
        self.fft_dims = tuple(range(-self.ndim, 0))

        # get device and array module
        device = get_device(self.kx)
        xp = device.xp
        dtype = self.kx.dtype

        self.xp = xp
        self.device = device
        self.dtype = dtype

        # initialize stack axes from sizes if needed
        if self.slice_axis is None and self.nslices is not None:
            self.slice_axis = xp.arange(self.nslices, dtype=int)
        if self.contrast_axis is None and self.ncontrasts is not None:
            self.contrast_axis = xp.arange(self.ncontrasts, dtype=int)
        if self.time_axis is None and self.nframes is not None:
            self.time_axis = xp.arange(self.nframes, dtype=int)

        # Initialize axis indices and expand axes
        axis_defs = [
            ("time_axis", 0),
            ("contrast_axis", 1),
            ("slice_axis", 2),
        ]
        axis_index_map = {}
        shape_template = [1, 1, 1]

        for name, idx in axis_defs:
            arr = getattr(self, name)
            if arr is not None:
                # Register axis index
                axis_index_map[name + "_index"] = idx
                # Expand array shape to be broadcast-compatible
                expanded_shape = list(shape_template)
                expanded_shape[idx] = arr.shape[0]
                setattr(self, name, arr.reshape(expanded_shape))

        for attr, val in axis_index_map.items():
            setattr(self, attr, val)

        # store stack lengths
        self.nframes = self.time_axis.max() + 1 if self.time_axis is not None else 1
        self.ncontrasts = (
            self.contrast_axis.max() + 1 if self.contrast_axis is not None else 1
        )
        self.nslices = self.slice_axis.max() + 1 if self.slice_axis is not None else 1

        # Ensure k-space arrays are ≥2D
        for name in ["kx", "ky"]:
            arr = getattr(self, name)
            if arr.ndim < 2:
                raise ValueError(f"{name} must be at least 2D, got shape {arr.shape}")
        if self.kz is not None and self.kz.ndim < 2:
            raise ValueError(f"kz must be at least 2D, got shape {self.kz.shape}")

        # Handle the case for ndim=3 with slice_axis and kz
        if self.ndim == 3:
            if self.slice_axis is not None and self.kz is not None:
                raise ValueError(
                    "For ndim=3, cannot provide both slice_axis and kz. Please provide one or the other."
                )

            if self.slice_axis is not None:
                self._hybrid_trajectory = True
                if self.kz is None:
                    self.nz = self.nslices
                    self.kz = (
                        _astype(self.slice_axis, self.kx.dtype) - 0.5 * self.nslices
                    ) / self.nslices
                    self.kz = self.kz[..., None, None]

        # Compute expected stack shape
        full_coords_shape = [
            self.nframes if self.time_axis is not None else 1,
            self.ncontrasts if self.contrast_axis is not None else 1,
            self.nslices if self.slice_axis is not None else 1,
        ]
        self.kx = _normalize_kn(self.kx, "kx", full_coords_shape)
        self.ky = _normalize_kn(self.ky, "ky", full_coords_shape)
        if self.kz is not None:
            self.kz = _normalize_kn(self.kz, "kz", full_coords_shape)

        # enforce homogeneous dtype/device
        self.to(device, dtype, xp)

        # default normalization
        self.scale_coords(self.nx, self.ny, self.nz)

    def info(self) -> str:
        """Return a human-readable summary of trajectory shape and axes."""
        lines = []

        # Optional Z dimension
        if self.kz is not None:
            lines.append("3D trajectory (kx, ky, kz)")
        else:
            lines.append("2D trajectory (kx, ky)")

        # K-space info
        shape = self.kx.shape  # All k-space arrays have same shape after init
        nshots, npts = shape[-2], shape[-1]
        lines.append(f"K-space shape: {shape} (shots={nshots}, pts={npts})")

        # Stack axis info
        stack_info = []
        if self.time_axis is not None:
            stack_info.append(f"frames={self.nframes}")
        if self.contrast_axis is not None:
            stack_info.append(f"contrasts={self.ncontrasts}")
        if self.slice_axis is not None:
            stack_info.append(f"slices={self.nslices}")
        if stack_info:
            lines.append("Stack dimensions: " + ", ".join(stack_info))
        else:
            lines.append("Stack dimensions: none")

        return "\n".join(lines)

    def __repr__(self):  # noqa
        return self.info()

    def __str__(self):  # noqa
        return self.info()

    def coords(self):  # noqa
        fourier_coords = self.fourier_coords(raveled=True)
        stack_indexes = self.stack_indexes(raveled=True)
        if self._hybrid_trajectory:
            return fourier_coords[..., :-1], stack_indexes
        else:
            return fourier_coords, stack_indexes

    @property
    def shape(self):  # noqa
        return (self.nframes, self.ncontrasts, self.nslices, self.nx, self.ny, self.nz)

    @property
    def grid_shape(self):  # noqa
        return (self.nx, self.ny, self.nz)

    @property
    def stack_shape(self):  # noqa
        return (self.nframes, self.ncontrasts, self.nslices)

    def fourier_coords(self, raveled: bool = False) -> NDArray:
        """
        Return spatial k-space coordinates.

        Parameters
        ----------
        raveled : bool, optional
            If true, ravel coordinates before stacking. The default is ``False``.

        Returns
        -------
        NDArray
            K-Space coordinates of shape ``(..., ndim)``.

        """
        xp = self.xp
        if self._hybrid_trajectory:
            self.kz = self.xp.tile(self, self.kz, self.kx.shape[-2:])
            self.kx = xp.broadcast_to(self.kx, self.kz.shape).copy()
            self.ky = xp.broadcast_to(self.ky, self.kz.shape).copy()

        # assemble spatial coordinates
        if self.kz is not None:
            if raveled:
                return xp.stack(
                    (self.kx.ravel(), self.ky.ravel(), self.kz.ravel()), axis=-1
                )
            return xp.stack((self.kx, self.ky, self.kz), axis=-1)
        if raveled:
            return xp.stack((self.kx.ravel(), self.ky.ravel()), axis=-1)
        return xp.stack((self.kx, self.ky), axis=-1)

    def stack_indexes(self, raveled: bool = False) -> NDArray:
        """
        Return stack k-space indexes.

        Parameters
        ----------
        raveled : bool, optional
            If true, ravel coordinates before stacking. The default is ``False``.

        Returns
        -------
        NDArray
            K-Space stack indexes of shape ``(..., ndim)``.

        """
        stack_axes = []
        if self.time_axis is not None:
            if raveled:
                stack_axes.append(self.time_axis.ravel())
            else:
                stack_axes.append(self.time_axis)
        if self.contrast_axis is not None:
            if raveled:
                stack_axes.append(self.contrast_axis.ravel())
            else:
                stack_axes.append(self.contrast_axis)
        if self.slice_axis is not None:
            if raveled:
                stack_axes.append(self.slice_axis.ravel())
            else:
                stack_axes.append(self.slice_axis)
        return self.xp.stack(stack_axes, axis=-1)

    def scale_coords(self, ax: float, ay: float | None = None, az: float | None = None):
        """
        Rescale coordinates to the given amplitude.

        Parameters
        ----------
        ax : float
            Amplitude value for x coordinates, i.e., ``kx`` will be
            rescaled between ``[-ax / 2, ax / 2]``.
        ay : float | None, optional
            Amplitude value for y coordinates, i.e., ``ky`` will be
            rescaled between ``[-ay / 2, ay / 2]``. The default is ``None``,
            i.e., assume ``ay = ax``.
        az : float | None, optional
            Amplitude value for z coordinates, i.e., ``kz`` will be
            rescaled between ``[-az / 2, az / 2]``. The default is ``None``,
            i.e., assume ``az = ax``.

        """
        ay = ax if ay is None else ay
        az = ax if az is None else az

        # rescale
        self.kx = ax * self.kx / abs(self.kx).max()
        self.ky = ay * self.ky / abs(self.ky).max()
        if self.kz is not None:
            self.kz = az * self.kz / abs(self.kz).max()

    def to(self, *args, **kwargs):  # noqa
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        array_module = kwargs.get("array_module", None)

        # check arg
        for arg in args:
            if (isinstance(arg, str) and ("cpu" in arg or "cuda" in arg)) or isinstance(
                arg, DeviceType
            ):
                device = arg
            elif isinstance(arg, str):
                if arg in ["numpy", "np"]:
                    array_module = np
                elif arg in ["cupy", "cp"]:
                    array_module = cp
                elif arg == "torch":
                    array_module = torch
            elif isinstance(arg, ModuleType):
                array_module = arg
            else:
                dtype = arg

        # replace default
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        xp = self.xp if array_module is None else array_module

        # enforce homogeneous trajectory
        self.kx = to_array_module(self.kx, xp, device)
        self.kx = _astype(self.kx, dtype)
        self.ky = to_array_module(self.ky, xp, device)
        self.ky = _astype(self.ky, dtype)
        if self.kz is not None:
            self.kz = to_array_module(self.kz, xp, device)
            self.kz = _astype(self.kz, dtype)

        # cast stack axes
        if self.slice_axis is None and self.nslices is not None:
            self.slice_axis = to_array_module(self.slice_axis, xp, device)
        if self.contrast_axis is None and self.ncontrasts is not None:
            self.contrast_axis = to_array_module(self.contrast_axis, xp, device)
        if self.time_axis is None and self.nframes is not None:
            self.time_axis = to_array_module(self.time_axis, xp, device)

        return self


# %% utils
def _astype(input: NDArray, dtype: DTypeLike):
    if isinstance(input, torch.Tensor):
        return input.to(NP2TORCH[dtype])
    else:
        return input.astype(dtype)


def _normalize_kn(kn, name, full_stack_shape):
    ndim_stack = len(full_stack_shape)

    # Pad with 1s to reach expected rank
    missing_dims = ndim_stack + 2 - kn.ndim
    if missing_dims > 0:
        kn = kn.reshape((1,) * missing_dims + kn.shape)

    # Check that all leading axes are either 1 or match expected length
    for i, (actual, expected) in enumerate(
        zip(kn.shape[:ndim_stack], full_stack_shape)
    ):
        if actual != expected and actual != 1:
            raise ValueError(
                f"{name}: leading dimension {i} = {actual}, expected {expected} or 1 (broadcastable)"
            )

    return kn
