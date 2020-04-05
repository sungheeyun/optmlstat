from abc import ABC, abstractmethod
from typing import Optional

from numpy import ndarray


class VectorOperatorBase(ABC):
    """
    The base class for vector operators.
    This represents an operator from n-dimensional vector space to m-dimensional vector space.
    """

    def __init__(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None) -> None:
        self.input_dim: Optional[int] = input_dim
        self.output_dim: Optional[int] = output_dim

    @abstractmethod
    def transform(self, input_array_1d: ndarray) -> ndarray:
        """
        Parameters
        ----------
        input_array_1d: ndarray
         1-d input array

        Returns
        -------
        return output_array_1d: ndarray
         1-d output array
        """
        pass
