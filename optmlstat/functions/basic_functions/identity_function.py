"""
identity function
"""

import numpy as np
from numpy import ndarray

from optmlstat.functions.function_base import FunctionBase


class IdentityFunction(FunctionBase):

    @property
    def is_twice_differentiable(self) -> bool:
        return True

    def _hessian(self, x_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __init__(self, num_vars: int) -> None:
        self.num_vars: int = num_vars

    def _jacobian(self, x_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def is_strictly_concave(self) -> bool:
        return False

    @property
    def is_concave(self) -> bool:
        return True

    @property
    def is_differentiable(self) -> bool:
        return True

    @property
    def conjugate(self) -> FunctionBase:
        raise NotImplementedError()

    def conjugate_arg(self, z_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def num_inputs(self) -> int:
        return self.num_vars

    @property
    def num_outputs(self) -> int:
        return self.num_vars

    @property
    def is_affine(self) -> bool:
        return True

    @property
    def is_strictly_convex(self) -> bool:
        return False

    @property
    def is_convex(self) -> bool:
        return True

    def _get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        return x_array_2d.copy()
