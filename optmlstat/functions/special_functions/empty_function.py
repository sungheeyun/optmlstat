"""
empty function, especially for instantiating dual problem class
"""

import numpy as np

from optmlstat.functions.function_base import FunctionBase


class EmptyFunction(FunctionBase):

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        self._num_inputs: int = num_inputs
        self._num_outputs: int = num_outputs

    def _get_y_values_2d(self, x_array_2d: np.ndarray) -> np.ndarray:
        return np.ones((x_array_2d.shape[0], self.num_outputs)) * np.nan

    def _hessian(self, x_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _jacobian(self, x_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def conjugate(self) -> FunctionBase:
        raise NotImplementedError()

    def conjugate_arg(self, z_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def is_affine(self) -> bool:
        return False

    @property
    def is_concave(self) -> bool:
        raise NotImplementedError()

    @property
    def is_convex(self) -> bool:
        return False

    @property
    def is_differentiable(self) -> bool:
        return False

    @property
    def is_strictly_concave(self) -> bool:
        raise NotImplementedError()

    @property
    def is_strictly_convex(self) -> bool:
        raise NotImplementedError()

    @property
    def is_twice_differentiable(self) -> bool:
        raise NotImplementedError()

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
    def num_inputs(self) -> int:
        return self._num_inputs

    @property
    def num_outputs(self) -> int:
        return self._num_outputs
