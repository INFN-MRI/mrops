"""Custom type hints for mrops."""

from typing import Union
import numpy as np

try:
    import cupy as cp
    CupyArray = cp.ndarray
except ImportError:
    CupyArray = None

try:
    import torch
    TorchTensor = torch.Tensor
except ImportError:
    TorchTensor = None

ArrayLike = Union[np.ndarray, CupyArray, TorchTensor]
