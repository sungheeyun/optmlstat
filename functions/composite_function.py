from typing import Optional, List

from numpy import ndarray

from optmlstat.functions.function_base import FunctionBase


class CompositeFunction(FunctionBase):
    """
    Composite function:

    f(x) = f_n(f_{n-1} ... f_1(x)))

    """

    def __init__(self, function_list: List[FunctionBase]) -> None:
        self.function_list: List[FunctionBase] = function_list

    def get_num_inputs(self) -> Optional[int]:
        if len(self.function_list) > 0:
            return self.function_list[0].get_num_inputs()
        else:
            return super(CompositeFunction, self).get_num_inputs()

    def get_num_outputs(self) -> Optional[int]:
        if len(self.function_list) > 0:
            return self.function_list[-1].get_num_outputs()

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        y_array_2d: ndarray = x_array_2d.copy()
        for function in self.function_list:
            y_array_2d = function.get_y_values_2d(y_array_2d)

        return y_array_2d
