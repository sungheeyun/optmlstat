from numpy.core._multiarray_umath import ndarray

from optmlstat.functions.function_base import FunctionBase


class IdentityFunction(FunctionBase):
    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        return x_array_2d.copy()
