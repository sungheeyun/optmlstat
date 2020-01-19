from typing import Callable, Iterable, Union, List, Optional

from numpy import ndarray, vectorize, vstack

from optmlstat.functions.function_base import FunctionBase


class ComponentWiseFunction(FunctionBase):
    """
    Component-wise function.

    f(x) = [f1(x1) f2(x2) ... fn(xn)]^T
    """
    def __init__(self, ufcn_or_list: Union[Callable, Iterable]) -> None:
        self.ufcn: Optional[Callable] = None
        self.ufcn_list: Optional[List[Callable]] = None
        self.vfcn: Optional[vectorize] = None
        self.vfcn_list: Optional[List[vectorize]] = None

        if isinstance(ufcn_or_list, Callable):
            self.ufcn = ufcn_or_list
            self.vfcn = vectorize(self.ufcn)
        elif isinstance(ufcn_or_list, Iterable):
            self.ufcn_list = list(ufcn_or_list)
            self.vfcn_list = [vectorize(ufcn) for ufcn in self.ufcn_list]
        else:
            assert False, ufcn_or_list.__class__

    def get_num_inputs(self) -> Optional[int]:
        if self.ufcn_list is None:
            return None
        else:
            return len(self.ufcn_list)

    def get_num_outputs(self) -> Optional[int]:
        return self.get_num_inputs()

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        if self.vfcn is None:
            return vstack([self.vfcn_list[idx](x_array_1d) for idx, x_array_1d in enumerate(x_array_2d.T)]).T
        else:
            return self.vfcn(x_array_2d)
