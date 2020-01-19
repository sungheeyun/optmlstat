from numpy import ndarray

from functions.function_base import FunctionBase


class IdentityFunction(FunctionBase):
    def __init__(self):
        super(IdentityFunction, self).__init__()

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        return x_array_2d.copy()
