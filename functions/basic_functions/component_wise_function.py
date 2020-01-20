from typing import Callable, Iterable, Union, List, Optional

from numpy import ndarray, vectorize, vstack

from functions.function_base import FunctionBase


class ComponentWiseFunction(FunctionBase):
    """
    Component-wise function.

    f(x) = [f1(x1) f2(x2) ... fn(xn)]^T
    """

    def __init__(self, unit_fcn_or_list: Union[Callable, Iterable]) -> None:
        self.unit_fcn: Optional[Callable] = None
        self.unit_fcn_list: Optional[List[Callable]] = None
        self.vectorize_fcn: Optional[vectorize] = None
        self.vectorize_fcn_list: Optional[List[vectorize]] = None

        if callable(unit_fcn_or_list):
            self.unit_fcn = unit_fcn_or_list
            self.vectorize_fcn = vectorize(self.unit_fcn)
        elif isinstance(unit_fcn_or_list, Iterable):
            self.unit_fcn_list = list(unit_fcn_or_list)
            self.vectorize_fcn_list = [vectorize(ufcn) for ufcn in self.unit_fcn_list]
        else:
            assert False, unit_fcn_or_list.__class__

        self._num_inputs: Optional[int] = None if self.unit_fcn_list is None else len(self.unit_fcn_list)

    @property
    def num_inputs(self) -> Optional[int]:
        return self._num_inputs

    @property
    def num_outputs(self) -> Optional[int]:
        return self._num_inputs

    @property
    def is_affine(self) -> Optional[bool]:
        return None

    @property
    def is_strictly_convex(self) -> Optional[bool]:
        return None

    @property
    def is_convex(self) -> Optional[bool]:
        return None

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        if self.vectorize_fcn_list is not None:
            return vstack([self.vectorize_fcn_list[idx](x_array_1d) for idx, x_array_1d in enumerate(x_array_2d.T)]).T
        elif self.vectorize_fcn is not None:
            return self.vectorize_fcn(x_array_2d)

        return ndarray((self.num_inputs, self.num_outputs))
