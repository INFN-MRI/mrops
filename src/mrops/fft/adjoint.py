"""
Adjoint FFT operator.
"""

from ..base_operator import MRIBaseOperator
from ..utils.typing import ArrayLike

class AdjointFFT(MRIBaseOperator):
    """
    Adjoint FFT operator for MRI encoding.
    """
    def __init__(self, shape: tuple, normalized: bool = True) -> None:
        self.shape = shape
        self.normalized = normalized

    def __call__(self, x: ArrayLike) -> ArrayLike:
        pass  # Placeholder for implementation
