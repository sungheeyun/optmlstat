from typing import List, Optional

from numpy import ndarray, all

from functions.function_base import FunctionBase


class CompositeFunction(FunctionBase):
    """
    Composite function:

    f(x) = f_n(f_{n-1} ... f_1(x)))

    """

    def __init__(self, function_list: List[FunctionBase]) -> None:
        assert function_list
        self.function_list: List[FunctionBase] = function_list

        self.shape_tuple: tuple = tuple([function.get_shape() for function in function_list])

        self._is_affine: Optional[bool] = None
        if all([function.is_affine for function in self.function_list]):
            self._is_affine = True

    @property
    def num_inputs(self) -> Optional[int]:
        return self.function_list[0].num_inputs

    @property
    def num_outputs(self) -> Optional[int]:
        return self.function_list[-1].num_outputs

    @property
    def is_affine(self) -> Optional[bool]:
        return self._is_affine

    @property
    def is_strictly_convex(self) -> Optional[bool]:
        return None

    @property
    def is_convex(self) -> Optional[bool]:
        return None

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        y_array_2d: ndarray = x_array_2d
        for function in self.function_list:
            y_array_2d = function.get_y_values_2d(y_array_2d)

        return y_array_2d

    def get_shape_tuple(self) -> tuple:
        return self.shape_tuple
