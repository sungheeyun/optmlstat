"""
composite of multiple functions
"""

import numpy as np
from numpy import ndarray, all

from optmlstat.functions.function_base import FunctionBase


class CompositeFunction(FunctionBase):
    """
    Composite function:

    f = f_n \circ f_{n-1} \circ ... \circ f_1  # noqa:W605
    """

    def __init__(self, function_list: list[FunctionBase]) -> None:
        assert function_list
        self.function_list: list[FunctionBase] = function_list

        self.shape_tuple: tuple = tuple([function.get_shape() for function in function_list])

        self._is_affine: bool = False
        if all([function.is_affine for function in self.function_list]):
            self._is_affine = True

    @property
    def maximal_point(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def maximal_value(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def maximum_point(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def maximum_value(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def minimal_point(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def minimal_value(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def minimum_point(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def minimum_value(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def is_twice_differentiable(self) -> bool:
        raise NotImplementedError()

    def _hessian(self, x_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def is_strictly_concave(self) -> bool:
        raise NotImplementedError()

    @property
    def is_concave(self) -> bool:
        raise NotImplementedError()

    @property
    def is_differentiable(self) -> bool:
        return bool(all([fcn.is_differentiable for fcn in self.function_list]))

    def _jacobian(self, x_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def conjugate(self) -> FunctionBase:
        raise NotImplementedError()

    def conjugate_arg(self, z_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

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
        raise NotImplementedError()

    @property
    def is_convex(self) -> bool:
        raise NotImplementedError()

    def _get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        y_array_2d: ndarray = x_array_2d
        for function in self.function_list:
            y_array_2d = function.get_y_values_2d(y_array_2d)

        return y_array_2d

    def get_shape_tuple(self) -> tuple:
        return self.shape_tuple
