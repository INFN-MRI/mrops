"""Backend utils."""

__all__ = [
    "Device",
    "cpu_device",
    "CUPY_AVAILABLE",
    "CUDA_AVAILABLE",
    "get_device",
    "get_array_module",
    "to_array_module",
    "to_device",
    "with_numpy",
    "with_numpy_cupy",
    "with_torch",
]

from types import ModuleType
from numpy.typing import NDArray

import numpy as _np
import torch as _torch

from mrinufft._array_compat import (
    with_numpy,
    with_numpy_cupy,
    with_torch,
    _to_interface,
)

CUPY_AVAILABLE = True
try:
    import cupy as _cp
    from cupy import cublas as _cublas

    _x = _cp.arange(4, dtype=_cp.complex64)
    _y = _cp.arange(4, dtype=_cp.complex64)
    _cublas.dotc(_x, _y)
except ImportError:
    CUPY_AVAILABLE = False

CUDA_AVAILABLE = CUPY_AVAILABLE or _torch.cuda.is_available()


class Device:
    """
    Device class.

    This class extends cupy.cuda.Device, with id > 0 representing the id_th GPU,
    and id = -1 representing CPU. cupy must be installed to use GPUs.

    The array module for the corresponding device can be obtained via .xp.
    Similar to cupy.cuda.Device, the Device object can be used as a context:

        >>> device = Device(2)
        >>> xp = device.xp  # xp is cupy.
        >>> with device:
        >>>     x = xp.array([1, 2, 3])
        >>>     x += 1

    Parameters
    ----------
    id_or_device : int | str | cupy.cuda.Device | torch.device | Device
        ``id = -1`` represents CPU, and others represents the id_th GPUs.
        Also accepts device as string following PyTorch conventions
        (e.g., ``cpu`` or ``cuda:0``).

    Attributes
    ----------
    id : int
        ``id = -1`` represents CPU, and others represents the id_th GPUs.

    """

    def __init__(self, id_or_device):
        if isinstance(id_or_device, int):
            id = id_or_device
        elif isinstance(id_or_device, str):
            id_or_device = _torch.device(id_or_device)
            if id_or_device.index is None:
                if id_or_device.type == "cuda":
                    id = 0
                else:
                    id = -1
            else:
                id = id_or_device.index
        elif CUPY_AVAILABLE and isinstance(id_or_device, _cp.cuda.Device):
            id = id_or_device.id
        elif isinstance(id_or_device, Device):
            id = id_or_device.id
        elif isinstance(id_or_device, _torch.device):
            if id_or_device.index is None:
                if id_or_device.type == "cuda":
                    id = 0
                else:
                    id = -1
            else:
                id = id_or_device.index
        else:
            raise ValueError(
                "Accepts int, str, cupy.cuda.Device, torch.device or Device, got {}".format(
                    id_or_device
                )
            )

        # check validity
        available_gpus = [
            _torch.device(i).index for i in range(_torch.cuda.device_count())
        ]
        if id >= 0 and id not in available_gpus:
            raise ValueError(
                f"Selected device id {id} not available (available devices: {[-1] + available_gpus})."
            )

        # context manager
        if CUPY_AVAILABLE:
            self.cpdevice = _cp.cuda.Device(id)

        self.id = id

    @property
    def xp(self):
        """module: numpy or cupy module for the device."""
        if self.id == -1:
            return _np
        return _cp

    @property
    def torch(self):
        """module: torch."""
        return _torch

    @property
    def _torch_device(self):
        if self.id == -1:
            return "cpu"
        else:
            return f"cuda:{self.id}"

    def __int__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, int):
            return self.id == other
        elif CUPY_AVAILABLE and isinstance(other, _cp.cuda.Device):
            return self.id == other.id
        elif isinstance(other, _torch.device):
            other = _torch.device(other).index
            other = -1 if other is None else other
            return self.id == other
        elif isinstance(other, Device):
            return self.id == other.id
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        if self.id == -1:
            return "<CPU Device>"

        return f"<CUDA Device {self.id}>"

    # context manager
    def use(self):
        """
        Use computing device.

        All operations after use() will use the device.
        """
        if self.id > 0:
            self.cpdevice.use()

    def __enter__(self):
        if self.id == -1:
            return None

        return self.cpdevice.__enter__()

    def __exit__(self, *args):
        if self.id == -1:
            pass
        else:
            self.cpdevice.__exit__()


cpu_device = Device(-1)


def get_array_module(input: NDArray) -> ModuleType:
    """
    Gets an appropriate module from :mod:`numpy`, :mod:`cupy` or :mod:`torch`.

    This is almost equivalent to :func:`cupy.get_array_module`. The differences
    are that this function can be used even if cupy is not available.

    Parameters
    ----------
    input : NDArray
        Input array.

    Returns
    -------
    module : ModuleType
        Output :mod:`torch`, :mod:`cupy` or :mod:`numpy` is returned based on ``input``.

    """
    if isinstance(input, _torch.Tensor):
        return _torch
    if CUPY_AVAILABLE:
        return _cp.get_array_module(input)
    return _np


def to_array_module(
    input: NDArray,
    array_module: ModuleType,
    device: int | str | _cp.cuda.Device | _torch.device | Device | None = None,
) -> NDArray:
    """
    Set an appropriate module from :mod:`numpy`, :mod:`cupy` or :mod:`torch`.

    Parameters
    ----------
    input : NDArray
        Input array.
    array_module : str | ModuleType
        Output module type.
    device :  int | str | cupy.cuda.Device | torch.device | Device | None, optional
        Output device. The default is ``None`` (same as input).

    Returns
    -------
    output : NDArray
        Output array with module: :mod:`torch`, :mod:`cupy` or :mod:`numpy`
        is returned based on ``input``. If specified, array is also transferred
        to ``device`` - by default, performs zero-copy transfer to desired
        array module on the same device.

    """
    if isinstance(array_module, str):
        if array_module == "numpy" or array_module == "np":
            array_module = _np
        if array_module == "cupy" or array_module == "cp":
            array_module = _cp
        if array_module == "torch":
            array_module = _torch
    if device is not None:
        device = Device(device)
        if array_module.__name__ == "torch":
            device = device._torch_device
        else:
            device = device.id
    return _to_interface(input, array_module, device)


def get_device(input: NDArray) -> Device:
    """
    Get Device from input array.

    Parameters
    ----------
    input : NDArray
        Input array.

    Returns
    -------
    device :  Device
        Computational Device.

    """
    if get_array_module(input) == _np:
        return cpu_device
    else:
        return Device(input.device)


def to_device(
    input: NDArray,
    device: int | str | _cp.cuda.Device | _torch.device | Device = cpu_device,
) -> NDArray:
    """
    Move input to device. Does not copy if same device.

    Parameters
    ----------
    input : NDArray
        Input array.
    device :  int | str | cupy.cuda.Device | torch.device | Device, optional
        Output device. The default is ``cpu_device``.

    Returns
    -------
    output : NDArray
        Output array placed in ``device``.

    """
    idevice = get_device(input)
    odevice = Device(device)

    if idevice == odevice:
        return input

    if odevice == cpu_device:
        if isinstance(input, _torch.Tensor):
            return input.numpy(force=True)
        return input.get()
    else:
        if isinstance(input, _torch.Tensor):
            return input.to(_torch.device(odevice.id))
        with odevice:
            return _cp.asarray(input)
