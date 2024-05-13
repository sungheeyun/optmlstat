"""
affine functions
"""

from logging import Logger, getLogger

import numpy as np
from numpy import ndarray

from optmlstat.functions.function_base import FunctionBase

logger: Logger = getLogger()


class AffineFunction(FunctionBase):
    """
    Affine function.
    """

    def jacobian(self, x_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __init__(self, slope_array_2d: ndarray, intercept_array_1d: ndarray) -> None:
        assert slope_array_2d.ndim == 2, slope_array_2d.ndim
        assert intercept_array_1d.ndim == 1, intercept_array_1d.ndim
        assert slope_array_2d.shape[1] == intercept_array_1d.size, (
            slope_array_2d.shape,
            intercept_array_1d.shape,
        )

        self._slope_array_2d: ndarray = slope_array_2d.copy()
        self._intercept_array_1d: ndarray = intercept_array_1d.copy()

    @property
    def slope_array_2d(self) -> ndarray:
        return self._slope_array_2d

    @property
    def intercept_array_1d(self) -> ndarray:
        return self._intercept_array_1d

    @property
    def coef_array_2d(self):
        return self.slope_array_2d

    @property
    def constant_array_1d(self):
        return self.intercept_array_1d

    @property
    def a_array_2d(self):
        return self.slope_array_2d

    @property
    def b_array_1d(self):
        return self.intercept_array_1d

    @property
    def num_inputs(self) -> int:
        return self.slope_array_2d.shape[0]

    @property
    def num_outputs(self) -> int:
        return self.slope_array_2d.shape[1]

    @property
    def is_affine(self) -> bool:
        return True

    @property
    def is_strictly_convex(self) -> bool:
        return False

    @property
    def is_convex(self) -> bool:
        return True

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
        assert False
        # TODO (3) should we defined non-function? :)
        return FunctionBase()

    def conjugate_arg(self, z_array_2d: ndarray) -> ndarray:
        assert False
        return ndarray(0)

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        logger.debug(x_array_2d.shape)
        logger.debug(self.slope_array_2d.shape)
        logger.debug(self.intercept_array_1d.shape)
        return x_array_2d.dot(self.slope_array_2d) + self.intercept_array_1d
