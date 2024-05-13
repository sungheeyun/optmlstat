"""
composite of multiple functions
"""

import numpy as np
from numpy import ndarray, all

from optmlstat.functions.function_base import FunctionBase


class CompositeFunction(FunctionBase):
    """
    Composite function:

    f(x) = f_n(f_{n-1} ... f_1(x)))

    """

    @property
    def is_strictly_concave(self) -> bool:
        raise NotImplementedError()

    @property
    def is_concave(self) -> bool:
        raise NotImplementedError()

    @property
    def is_differentiable(self) -> bool:
        return all([fcn.is_differentiable for fcn in self.function_list])

    def jacobian(self, x_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def conjugate(self) -> FunctionBase:
        raise NotImplementedError()

    def conjugate_arg(self, z_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __init__(self, function_list: list[FunctionBase]) -> None:
        assert function_list
        self.function_list: list[FunctionBase] = function_list

        self.shape_tuple: tuple = tuple([function.get_shape() for function in function_list])

        self._is_affine: bool | None = None
        if all([function.is_affine for function in self.function_list]):
            self._is_affine = True

    @property
    def num_inputs(self) -> int:
        return self.function_list[0].num_inputs

    @property
    def num_outputs(self) -> int:
        return self.function_list[-1].num_outputs

    @property
    def is_affine(self) -> bool:
        return self._is_affine

    @property
    def is_strictly_convex(self) -> bool:
        return None

    @property
    def is_convex(self) -> bool:
        return None

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        y_array_2d: ndarray = x_array_2d
        for function in self.function_list:
            y_array_2d = function.get_y_values_2d(y_array_2d)

        return y_array_2d

    def get_shape_tuple(self) -> tuple:
        return self.shape_tuple
