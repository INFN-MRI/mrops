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
    A flexible and extensible representation of a k-space trajectory for MRI.

    This container supports optional stack dimensions (time, contrast, and slice) and
    2D/3D spatial configurations.

    This class handles the storage, normalization, and broadcasting of k-space
    coordinates (``kx``, ``ky``, and optionally ``kz``) along with their associated
    stack dimensions (``time_axis``, ``contrast_axis``, ``slice_axis``). It supports
    both standard and hybrid trajectories, with automatic expansion of stack
    axes and validation of coordinate shapes.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (2 or 3).
    nx : int
        Grid size along the x-axis.
    ny : int
        Grid size along the y-axis.
    kx : NDArray
        X-coordinates of k-space samples (≥2D).
    ky : NDArray
        Y-coordinates of k-space samples (≥2D).
    nz : int, optional
        Grid size along the z-axis (required if ``kz`` is provided).
    kz : NDArray, optional
        Z-coordinates of k-space samples.
    slice_axis : NDArray, optional
        1D index array for slices.
    contrast_axis : NDArray, optional
        1D index array for contrast types.
    time_axis : NDArray, optional
        1D index array for time frames.
    nslices : int, optional
        Number of slices. Required if ``slice_axis` ' is not provided.
    ncontrasts : int, optional
        Number of contrast types. Required if ` 'contrast_axis`` is not provided.
    nframes : int, optional
        Number of time frames. Required if ` 'time_axis`` is not provided.

    Attributes
    ----------
    coords_and_indexes : tuple
        Returns broadcasted k-space coordinates and corresponding stack indices.
    shape : tuple
        Full shape of the data implied by trajectory and stack axes.
    stack_shape : tuple
        Shape of stack dimensions ``(frames, contrasts, slices)``.
    grid_shape : tuple
        Shape of spatial grid ``(ny, nx)`` or ``(nz, ny, nx)``.
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

        # Initialize stack axes from sizes if needed
        if self.slice_axis is None and self.nslices is not None:
            self.slice_axis = xp.arange(self.nslices, dtype=int)
        if self.contrast_axis is None and self.ncontrasts is not None:
            self.contrast_axis = xp.arange(self.ncontrasts, dtype=int)
        if self.time_axis is None and self.nframes is not None:
            self.time_axis = xp.arange(self.nframes, dtype=int)

        # Determine which stack axes are active and assign dynamic indices
        active_axes = []
        if self.time_axis is not None:
            active_axes.append("time_axis")
        if self.contrast_axis is not None:
            active_axes.append("contrast_axis")
        if self.slice_axis is not None:
            active_axes.append("slice_axis")

        # Compute broadcast shapes and assign dynamic indices
        axis_defs = {
            "time_axis": 0,
            "contrast_axis": 1,
            "slice_axis": 2,
        }
        shape_template = [1, 1, 1]
        axis_index_map = {}

        for i, name in enumerate(active_axes):
            arr = getattr(self, name)
            if len(arr.shape) != 1:
                raise ValueError(f"{name} must be at most 1D, got shape {arr.shape}")
            shape = list(shape_template)
            shape[axis_defs[name]] = arr.shape[
                0
            ]  # Expand only in the correct dimension
            setattr(self, name, arr.reshape(shape))  # Explicit reshape
            axis_index_map[f"{name}_index"] = i  # Assign dynamic index

        # Assign computed indices (e.g., self.time_axis_index, self.slice_axis_index, etc.)
        for attr, val in axis_index_map.items():
            setattr(self, attr, val)

        # Store stack lengths
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

        # Handle ndim=3 with mutual exclusivity for slice_axis and kz
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
                    self.kz = self.kz[..., None, None]  # Add singleton dimensions

        # Compute expected stack shape
        full_coords_shape = [self.nframes, self.ncontrasts, self.nslices]
        self.kx = _normalize_kn(self.kx, "kx", full_coords_shape)
        self.ky = _normalize_kn(self.ky, "ky", full_coords_shape)
        if self.kz is not None:
            self.kz = _normalize_kn(self.kz, "kz", full_coords_shape)

        # Enforce homogeneous dtype/device
        self.to(device, dtype, xp)

        # Default normalization
        self.scale_coords(self.nx, self.ny, self.nz)

        # Lazy cache
        self._indexes = None
        self._coords = None

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

    @property
    def coords_and_indexes(self):  # noqa
        if self._indexes is not None:
            return self._coords, self._indexes
        xp = self.xp

        # Build trajectory
        kz = None
        if self._hybrid_trajectory:
            kz = self.xp.tile(self.kz, self.kx.shape[-2:])
            ky = xp.broadcast_to(self.ky, kz.shape).copy()
            kx = xp.broadcast_to(self.kx, kz.shape).copy()

            # Expand across time
            if self.time_axis is not None:
                kz = xp.repeat(kz, self.nframes, axis=0)
                ky = xp.repeat(ky, self.nframes, axis=0)
                kx = xp.repeat(kx, self.nframes, axis=0)

            # Expand across contrasts
            if self.contrast_axis is not None:
                kz = xp.repeat(kz, self.ncontrasts, axis=1)
                ky = xp.repeat(ky, self.ncontrasts, axis=1)
                kx = xp.repeat(kx, self.ncontrasts, axis=1)
        else:
            if self.ndim == 3:
                kz = self.kz
            ky = self.ky
            kx = self.kx

            # Expand across time
            if self.time_axis is not None:
                if self.ndim == 3:
                    kz = xp.repeat(kz, self.nframes, axis=0)
                ky = xp.repeat(ky, self.nframes, axis=0)
                kx = xp.repeat(kx, self.nframes, axis=0)

            # Expand across contrasts
            if self.contrast_axis is not None:
                if self.ndim == 3:
                    kz = xp.repeat(kz, self.ncontrasts, axis=1)
                ky = xp.repeat(ky, self.ncontrasts, axis=1)
                kx = xp.repeat(kx, self.ncontrasts, axis=1)

            # Expand across contrasts
            if self.slice_axis is not None and self.ndim == 2:
                ky = xp.repeat(ky, self.nslices, axis=2)
                kx = xp.repeat(kx, self.nslices, axis=2)

        if kz is not None:
            self._coords = xp.stack((kx, ky, kz), axis=-1)
        else:
            self._coords = xp.stack((kx, ky), axis=-1)

        # Build indexes
        self._indexes = []
        if self.time_axis is not None:
            self._indexes.append(
                xp.broadcast_to(self.time_axis[..., None, None], kx.shape)
            )

        # Expand across contrasts
        if self.contrast_axis is not None:
            self._indexes.append(
                xp.broadcast_to(self.contrast_axis[..., None, None], kx.shape)
            )

        # Expand across contrasts
        if self.slice_axis is not None:
            self._indexes.append(
                xp.broadcast_to(self.slice_axis[..., None, None], kx.shape)
            )
        if self._indexes:
            self._indexes = xp.stack(self._indexes, axis=-1)
            self._indexes = self._indexes.reshape(-1, self._indexes.shape[-1])

        # Reshape coords
        self._coords = self._coords.reshape(-1, self._coords.shape[-1])

        return self._coords, self._indexes

    @property
    def shape(self):  # noqa
        if self._hybrid_trajectory or self.ndim == 2:
            shape = [self.nframes, self.ncontrasts, self.nslices, self.ny, self.nx]
        else:
            shape = [self.nframes, self.ncontrasts, self.nz, self.ny, self.nx]
        return tuple([sz for sz in shape if sz != 1])

    @property
    def grid_shape(self):  # noqa
        if self._hybrid_trajectory or self.ndim == 2:
            return (self.ny, self.nx)
        else:
            return (self.nz, self.ny, self.nx)

    @property
    def stack_shape(self):  # noqa
        shape = [self.nframes, self.ncontrasts, self.nslices]
        return tuple([sz for sz in shape if sz != 1])

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
