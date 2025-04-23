"""Trajectory container."""

__all__ = ["Trajectory"]

from dataclasses import dataclass
from types import ModuleType
from numpy.typing import NDArray, DTypeLike

import numpy as np
import torch

from ..backend import get_device, to_array_module, CUPY_AVAILABLE, Device

DeviceType = int | str | torch.device | Device
if CUPY_AVAILABLE:
    import cupy as cp

    DeviceType = DeviceType | cp.cuda.Device


@dataclass
class Trajectory:
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

    def __post_init__(self):
        self.fft_dims = tuple(range(-self.ndim, 0))

        # get device and array module
        device = get_device(self.kx)
        xp = device.xp
        dtype = self.kx.dtype

        self.xp = xp
        self.device = device
        self.dtype = dtype

        # initialize stack axes
        if self.slice_axis is None and self.nslices is not None:
            self.slice_axis = np.arange(self.nslices, dtype=int)
        if self.contrast_axis is None and self.ncontrasts is not None:
            self.contrast_axis = np.arange(self.ncontrasts, dtype=int)
        if self.time_axis is None and self.nframes is not None:
            self.time_axis = np.arange(self.nframes, dtype=int)

        # enforce homgeneous array module, dtype and device
        self.to(device, dtype, xp)

        # initialize z axis index in coordinates matrix
        if self.slice_axis is not None:
            self.slice_axis_index = 0

        # initialize contrast axis index in coordinates matrix
        if self.contrast_axis is not None:
            self.contrast_axis_index = 0

            # move z axis index by one (contrasts, z)
            if self.slice_axis is not None:
                self.slice_axis_index += 1

        # initialize frames axis index in coordinates matrix
        if self.time_axis is not None:
            self.time_axis_index = 0

            # move contrast axis index by one (frames, contrasts)
            if self.contrast_axis is not None:
                self.contrast_axis_index += 1

            # move z axis index by one (frames, ..., z)
            if self.slice_axis is not None:
                self.slice_axis_index += 1

        # default normalization for coordinates is (-0.5 * shape[n], 0.5 * shape[n])
        self.scale_coords(self.nx, self.ny, self.nz)

    def scale_coords(self, ax, ay=None, az=None):
        ay = ax if ay is None else ay
        az = ax if az is None else az

        # rescale
        self.kx = ax * self.kx / abs(self.kx).max()
        self.ky = ay * self.ky / abs(self.ky).max()
        if self.kz is not None:
            self.kz = az * abs(self.kz).max()

    def to(self, *args, **kwargs):
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        array_module = kwargs.get("array_module", None)

        # check arg
        for arg in args:
            if (
                isinstance(arg, str)
                and ("cpu" in arg or "cuda" in arg)
                or isinstance(arg, DeviceType)
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

    # @property
    # def index_matrix(self) -> NDArray:
    #     nshots, npts = self.kx.shape[-2:]
    #     ncoords = nshots * npts
    #     nstack = int(np.prod(self.kx.shape[:-2]))

    #     if self.stack_coords is not None:
    #         stack_labels = self.stack_coords.reshape(-1)
    #     else:
    #         stack_labels = np.arange(nstack)

    #     stack_idx = np.repeat(stack_labels, ncoords)
    #     coord_idx = np.tile(np.arange(ncoords), len(stack_labels))

    #     return np.stack([stack_idx, coord_idx], axis=-1)

    # @property
    # def value_matrix(self) -> NDArray:
    #     coords = [self.ky, self.kx]
    #     if self.kz is not None:
    #         coords.insert(0, self.kz)

    #     flat_coords = [np.reshape(c, (-1,)) for c in coords]
    #     stacked = np.stack(flat_coords, axis=-1)

    #     if self.stack_coords is not None:
    #         repeated_stack = np.repeat(self.stack_coords.reshape(-1), coords[0].shape[-1])
    #         stacked = np.concatenate([repeated_stack[:, None], stacked], axis=-1)

    #     return stacked

    # @classmethod
    # def from_components(
    #     cls,
    #     kx: NDArray,
    #     ky: NDArray,
    #     kz: Optional[NDArray] = None,
    #     time_axis: Optional[NDArray] = None,
    #     contrast_axis: Optional[NDArray] = None,
    #     nz: Optional[int] = None,
    #     grid_shape: Tuple[int, ...] = (256, 256),
    #     fft_dims: Tuple[int, ...] = (0, 1),
    # ):
    #     stack_coords = np.arange(nz).reshape(nz, 1, 1) if nz is not None else None

    #     return cls(
    #         kx=kx,
    #         ky=ky,
    #         kz=kz,
    #         time_axis=time_axis,
    #         contrast_axis=contrast_axis,
    #         stack_coords=stack_coords,
    #         grid_shape=grid_shape,
    #         fft_dims=fft_dims,
    #     )


def _astype(input: NDArray, dtype: DTypeLike):
    if isinstance(input, torch.Tensor):
        return input.to(dtype)
    else:
        return input.astype(dtype)
