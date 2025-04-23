"""Base container."""

__all__ = ["ArrayContainer"]

import numpy as np

from numpy.typing import NDArray, DTypeLike

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None


class ArrayContainer:
    backend: str
    device: str | None = None
    dtype: DTypeLike | str | None = None

    def _detect_backend(self, array: NDArray):
        """Detect backend based on array type."""
        if torch and isinstance(array, torch.Tensor):
            return "torch"
        elif cp and isinstance(array, cp.ndarray):
            return "cupy"
        return "numpy"

    def _convert(self, array: NDArray) -> NDArray:
        """Convert array to the specified backend, dtype, and device."""
        if array is None:
            return None
        if self.backend == "torch":
            return torch.as_tensor(array, dtype=self.dtype, device=self.device)
        elif self.backend == "cupy":
            with self.device:
                return cp.asarray(array, dtype=self.dtype)
        return np.asarray(array, dtype=self.dtype)

    def _arange(self, n: int, axis_index: int):
        """Generate range and reshape based on axis_index."""
        shape = [1] * 5
        shape[axis_index] = n
        if self.backend == "torch":
            return torch.arange(n, dtype=self.dtype, device=self.device).reshape(shape)
        elif self.backend == "cupy":
            with self.device:
                return cp.arange(n, dtype=self.dtype).reshape(shape)
        return np.arange(n, dtype=self.dtype).reshape(shape)

    def _check_shapes(self, *arrays: NDArray):
        """Ensure all arrays have matching backend, dtype, and device."""
        arrays = [a for a in arrays if a is not None]
        ref_backend = self.backend
        ref_dtype = self.dtype
        ref_device = self.device

        for a in arrays:
            if self._detect_backend(a) != ref_backend:
                raise TypeError("Backend mismatch in stack attributes.")
            if hasattr(a, "dtype") and str(a.dtype) != str(ref_dtype):
                raise TypeError("Dtype mismatch in stack attributes.")
            if ref_backend == "torch" and self.device is not None:
                if str(a.device) != str(ref_device):
                    raise TypeError("Device mismatch in stack attributes.")
            elif ref_backend == "cupy" and self.device is not None:
                pass  # Assume cupy does not need device handling here

    def to(self, backend: str, dtype: DTypeLike = None, device: Device = None):
        """Convert all relevant arrays to the given backend, dtype, and device."""
        self.backend = backend
        self.device = device
        self.dtype = dtype

        self.kx = self._convert(self.kx)
        self.ky = self._convert(self.ky)
        self.kz = self._convert(self.kz)
        self.contrast_axis = self._convert(self.contrast_axis)
        self.time_axis = self._convert(self.time_axis)

        self._check_shapes(
            self.kx, self.ky, self.kz, self.contrast_axis, self.time_axis
        )
