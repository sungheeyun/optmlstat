"""
component-wise function
"""

from typing import Callable, Iterable

import numpy as np
from numpy import ndarray, vectorize, vstack

from optmlstat.functions.function_base import FunctionBase


class ComponentWiseFunction(FunctionBase):
    """
    Component-wise function.

    f(x) = [f1(x1) f2(x2) ... fn(xn)]^T
    """

    @property
    def is_strictly_concave(self) -> bool:
        raise NotImplementedError()

    @property
    def is_concave(self) -> bool:
        raise NotImplementedError()

    @property
    def is_differentiable(self) -> bool:
        raise NotImplementedError()

    def jacobian(self, x_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def conjugate(self) -> FunctionBase:
        raise NotImplementedError()

    def conjugate_arg(self, z_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __init__(self, unit_fcn_or_list: Callable | Iterable) -> None:
        self.unit_fcn: Callable | None = None
        self.unit_fcn_list: list[Callable] | None = None
        self.vectorize_fcn: vectorize | None = None
        self.vectorize_fcn_list: list[vectorize] | None = None

        if callable(unit_fcn_or_list):
            self.unit_fcn = unit_fcn_or_list
            self.vectorize_fcn = vectorize(self.unit_fcn)
        elif isinstance(unit_fcn_or_list, Iterable):
            self.unit_fcn_list = list(unit_fcn_or_list)
            self.vectorize_fcn_list = [vectorize(ufcn) for ufcn in self.unit_fcn_list]
        else:
            assert False, unit_fcn_or_list.__class__

        self._num_inputs: int | None = (
            None if self.unit_fcn_list is None else len(self.unit_fcn_list)
        )

    @property
    def num_inputs(self) -> int:
        return self._num_inputs  # type:ignore

    @property
    def num_outputs(self) -> int:
        return self._num_inputs  # type:ignore

    @property
    def is_affine(self) -> bool:
        raise NotImplementedError()

    @property
    def is_strictly_convex(self) -> bool:
        raise NotImplementedError()

    @property
    def is_convex(self) -> bool:
        raise NotImplementedError()

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        if self.vectorize_fcn_list is not None:
            return vstack(
                [
                    self.vectorize_fcn_list[idx](x_array_1d)
                    for idx, x_array_1d in enumerate(x_array_2d.T)
                ]
            ).T
        elif self.vectorize_fcn is not None:
            return self.vectorize_fcn(x_array_2d)

        return ndarray((self.num_inputs, self.num_outputs))
