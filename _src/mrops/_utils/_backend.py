"""Backend utilities."""

__all__ = [
    "CUPY_AVAILABLE",
    "AUTOGRAD_AVAILABLE",
    "with_numpy_cupy",
    "with_numpy",
    "with_torch",
    "get_array_module",
    "get_device",
]

from numpy.typing import NDArray
import numpy as np

from mrinufft._array_compat import (
    get_array_module,
    with_numpy,
    with_numpy_cupy,
    with_torch,
    CUPY_AVAILABLE,
    AUTOGRAD_AVAILABLE,
)

if CUPY_AVAILABLE:
    import cupy as cp


class Device(object):
    """
    Device class.

    This class extends cupy.Device, with id > 0 representing the id_th GPU,
    and id = -1 representing CPU. cupy must be installed to use GPUs.

    The array module for the corresponding device can be obtained via .xp.
    Similar to cupy.Device, the Device object can be used as a context:

        >>> device = Device(2)
        >>> xp = device.xp  # xp is cupy.
        >>> with device:
        >>>     x = xp.array([1, 2, 3])
        >>>     x += 1

    Parameters
    ----------
    id_or_device : int | Device | cupy.cuda.Device
        id > 0 represents the corresponding GPUs, and id = -1 represents CPU.

    Attributes
    ----------
    id : int
        id = -1 represents CPU, and others represents the id_th GPUs.

    """

    def __init__(self, id_or_device):
        if isinstance(id_or_device, int):
            id = id_or_device
        elif isinstance(id_or_device, Device):
            id = id_or_device.id
        elif CUPY_AVAILABLE and isinstance(id_or_device, cp.cuda.Device):
            id = id_or_device.id
        else:
            raise ValueError(
                f"Accepts int, Device or cupy.cuda.Device, got {id_or_device}"
            )

        if id != -1:
            if CUPY_AVAILABLE:
                self.cpdevice = cp.cuda.Device(id)
            else:
                raise ValueError(f"cupy not installed, but set device {id}.")

        self.id = id

    @property
    def xp(self):
        """module: numpy or cupy module for the device."""
        if self.id == -1:
            return np

        return cp

    def use(self):
        """
        Use computing device.

        All operations after use() will use the device.
        """
        if self.id > 0:
            self.cpdevice.use()

    def __int__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, int):
            return self.id == other
        elif isinstance(other, Device):
            return self.id == other.id
        elif CUPY_AVAILABLE and isinstance(other, cp.cuda.Device):
            return self.id == other.id
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __enter__(self):
        if self.id == -1:
            return None

        return self.cpdevice.__enter__()

    def __exit__(self, *args):
        if self.id == -1:
            pass
        else:
            self.cpdevice.__exit__()

    def __repr__(self):
        if self.id == -1:
            return "<CPU Device>"

        return self.cpdevice.__repr__()


cpu_device = Device(-1)


def get_device(input: NDArray) -> Device:
    """
    Get Device from input array.

    Parameters
    ----------
    input : NDArray
        Input array.

    Returns
    -------
    Device
        Device corresponding to input.

    """
    if get_array_module(input) == np:
        return cpu_device
    else:
        return Device(array.device)
