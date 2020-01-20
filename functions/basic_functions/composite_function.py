from typing import List

from numpy import ndarray

from functions.function_base import FunctionBase


class CompositeFunction(FunctionBase):
    """
    Composite function:

    f(x) = f_n(f_{n-1} ... f_1(x)))

    """

    def __init__(self, function_list: List[FunctionBase]) -> None:
        assert function_list
        self.function_list: List[FunctionBase] = function_list

        super(CompositeFunction, self).__init__(
            self.function_list[0].get_num_inputs(), self.function_list[-1].get_num_outputs()
        )

        self.shape_tuple: tuple = tuple([function.get_shape() for function in function_list])

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        y_array_2d: ndarray = x_array_2d
        for function in self.function_list:
            y_array_2d = function.get_y_values_2d(y_array_2d)

        return y_array_2d

    def get_shape_tuple(self) -> tuple:
        return self.shape_tuple
