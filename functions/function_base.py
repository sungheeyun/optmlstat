from abc import abstractmethod
from typing import Optional, Tuple

from numpy import ndarray, array

from basic_modueles.class_base import OptMLStatClassBase


class FunctionBase(OptMLStatClassBase):
    """
    The base class for function classes.
    """

    def __init__(self, num_inputs: Optional[int] = None, num_outputs: Optional[int] = None):
        self.num_inputs: Optional[int] = num_inputs
        self.num_outputs: Optional[int] = num_outputs

        # True is the function is a convex function.
        # If self.num_outputs is greater than 1,
        # it is True if and only if all self.num_outputs functions are convex functions.
        self.is_convex: bool = False

        self.is_affine: bool = False

    def get_shape(self) -> Tuple[Optional[int], Optional[int]]:
        return self.num_inputs, self.num_outputs

    def check_x_array_dimension(self, x_array_2d: ndarray) -> None:
        if self.num_inputs is not None:
            assert x_array_2d.shape[1] == self.num_inputs

    def check_y_array_dimension(self, y_array_2d: ndarray) -> None:
        if self.num_outputs is not None:
            assert y_array_2d.shape[1] == self.num_outputs

    def get_y_values_2d_from_x_values_1d(self, x_array_1d: ndarray) -> ndarray:
        x_array_2d: ndarray = array([x_array_1d]).T
        return self.get_y_values_2d(x_array_2d)

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
