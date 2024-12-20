"""
Base class for MRI operators.
"""

from abc import ABC, abstractmethod
from .utils.typing import ArrayLike

class MRIBaseOperator(ABC):
    """
    Abstract base class for MRI encoding operators.
    """

    @abstractmethod
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """
        Apply the operator.
        """
        pass
