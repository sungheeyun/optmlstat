from abc import ABC, abstractmethod
from typing import Optional, Tuple

from numpy import ndarray, array


class FunctionBase(ABC):
    """
    Base class for function classes.
    """

    def __init__(self, num_inputs: Optional[int] = None, num_outputs: Optional[int] = None):
        self.num_inputs: Optional[int] = num_inputs
        self.num_outputs: Optional[int] = num_outputs

    def get_num_inputs(self) -> Optional[int]:
        return self.num_inputs

    def get_num_outputs(self) -> Optional[int]:
        return self.num_outputs

    def get_shape(self) -> Tuple[Optional[int], Optional[int]]:
        return self.get_num_inputs(), self.get_num_outputs()

    def check_x_array_dimension(self, x_array_2d: ndarray) -> None:
        if self.get_num_inputs() is not None:
            assert x_array_2d.shape[1] == self.get_num_inputs()

    def check_y_array_dimension(self, y_array_2d: ndarray) -> None:
        if self.get_num_outputs() is not None:
            assert y_array_2d.shape[1] == self.get_num_outputs()

    def get_y_values_2d_from_x_values_1d(self, x_array_1d: ndarray) -> ndarray:
        x_array_2d: ndarray = array([x_array_1d]).T
        return self.get_y_values_2d(x_array_2d)

    @abstractmethod
    def is_convex_function(self) -> bool:
        """
        Returns true is the function is a convex function.
        If the number of outputs is greater than 1,
        it means each of M functions are convex functions.
        """
        pass

    @abstractmethod
    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        """
        Returns y values for given x values.

        Parameters
        ----------
        x_array_2d:
          N-by-n array representing x.

        Returns
        -------
        y_array_2d:
          N-by-m array representing y.
        """
        pass
