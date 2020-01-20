from numpy import ndarray

from functions.function_base import FunctionBase


class AffineFunction(FunctionBase):
    """
    Affine function.
    """

    def __init__(self, slope_array_2d: ndarray, intercept_array_1d: ndarray) -> None:
        assert slope_array_2d.ndim == 2, slope_array_2d.ndim
        assert intercept_array_1d.ndim == 1, intercept_array_1d.ndim
        assert slope_array_2d.shape[1] == intercept_array_1d.size, (slope_array_2d.shape, intercept_array_1d.shape)

        self.slope_array_2d: ndarray = slope_array_2d.copy()
        self.intercept_array_1d: ndarray = intercept_array_1d.copy()

        super(AffineFunction, self).__init__(self.slope_array_2d.shape[0], self.slope_array_2d.shape[1])

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        return x_array_2d.dot(self.slope_array_2d) + self.intercept_array_1d

    def is_convex_function(self) -> bool:
        return True
