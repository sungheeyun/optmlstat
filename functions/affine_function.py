from typing import Optional

from numpy import ndarray

from optmlstat.functions.function_base import FunctionBase


class AffineFunction(FunctionBase):
    """
    Affine function.
    """

    def __init__(self, slope_array: ndarray, intercept_array: ndarray) -> None:
        assert slope_array.ndim == 2, slope_array.ndim
        assert intercept_array.ndim == 1, intercept_array.ndim
        assert slope_array.shape[1] == intercept_array.size, (slope_array.shape, intercept_array.shape)

        self.slope_array: ndarray = slope_array.copy()
        self.intercept_array: ndarray = intercept_array

    def get_num_outputs(self) -> Optional[int]:
        return self.slope_array.shape[1]

    def get_num_inputs(self) -> Optional[int]:
        return self.slope_array.shape[0]

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        return x_array_2d.dot(self.slope_array) + self.intercept_array
