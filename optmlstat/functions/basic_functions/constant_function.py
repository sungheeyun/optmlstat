"""
constant functions
"""

from logging import Logger, getLogger

import numpy as np

from optmlstat.functions.exceptions import InfiniteNumberOfSolutionsException
from optmlstat.functions.function_base import FunctionBase

logger: Logger = getLogger()


class ConstantFunction(FunctionBase):
    """
    constant function.
    """

    def __init__(self, y_1d: np.ndarray, num_inputs: int) -> None:
        self._y_1d: np.ndarray = y_1d.copy()
        self._num_inputs: int = num_inputs

    def _get_y_values_2d(self, x_2d: np.ndarray) -> np.ndarray:
        return self._y_1d[np.newaxis, :].repeat(x_2d.shape[0], axis=0)

    def _hessian(self, x_2d: np.ndarray) -> np.ndarray:
        return np.zeros((x_2d.shape[0], self.num_outputs, self.num_inputs, self.num_inputs))

    def _jacobian(self, x_2d: np.ndarray) -> np.ndarray:
        return np.zeros((x_2d.shape[0], self.num_outputs, self.num_inputs))

    @property
    def conjugate(self) -> FunctionBase:
        raise NotImplementedError()

    def conjugate_arg(self, z_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def is_affine(self) -> bool:
        return True

    @property
    def is_concave(self) -> bool:
        return True

    @property
    def is_convex(self) -> bool:
        return True

    @property
    def is_differentiable(self) -> bool:
        return True

    @property
    def is_strictly_concave(self) -> bool:
        return False

    @property
    def is_strictly_convex(self) -> bool:
        return False

    @property
    def is_twice_differentiable(self) -> bool:
        return True

    @property
    def maximal_point(self) -> np.ndarray:
        return self.maximum_point[np.newaxis, :]

    @property
    def maximal_value(self) -> np.ndarray:
        return self.maximum_value[np.newaxis, :]

    @property
    def maximum_point(self) -> np.ndarray:
        raise InfiniteNumberOfSolutionsException()

    @property
    def maximum_value(self) -> np.ndarray:
        return self._y_1d

    @property
    def minimal_point(self) -> np.ndarray:
        return self.minimum_point[np.newaxis, :]

    @property
    def minimal_value(self) -> np.ndarray:
        return self.minimum_value[np.newaxis, :]

    @property
    def minimum_point(self) -> np.ndarray:
        raise InfiniteNumberOfSolutionsException()

    @property
    def minimum_value(self) -> np.ndarray:
        return self._y_1d

    @property
    def num_inputs(self) -> int:
        return self._num_inputs

    @property
    def num_outputs(self) -> int:
        return self._y_1d.size
