"""
Forward coil sensitivity operator.
"""

from ..base_operator import MRIBaseOperator
from ..utils.typing import ArrayLike

class ForwardCoilSensitivity(MRIBaseOperator):
    """
    Forward coil sensitivity operator for MRI encoding.
    """
    def __init__(self, sensitivities: ArrayLike) -> None:
        self.sensitivities = sensitivities

    def __call__(self, x: ArrayLike) -> ArrayLike:
        pass  # Placeholder for implementation
