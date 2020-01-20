from logging import Logger, getLogger
from typing import Optional

from numpy import ndarray

from functions.function_base import FunctionBase


logger: Logger = getLogger()


class AffineFunction(FunctionBase):
    """
    Affine function.
    """

    def __init__(self, slope_array_2d: ndarray, intercept_array_1d: ndarray) -> None:
        assert slope_array_2d.ndim == 2, slope_array_2d.ndim
        assert intercept_array_1d.ndim == 1, intercept_array_1d.ndim
        assert slope_array_2d.shape[1] == intercept_array_1d.size, (slope_array_2d.shape, intercept_array_1d.shape)

        self.slope_array_2d: ndarray = slope_array_2d.copy()
        self.intercept_array_1d: ndarray = intercept_array_1d.copy()

    @property
    def num_inputs(self) -> Optional[int]:
        return self.slope_array_2d.shape[0]

    @property
    def num_outputs(self) -> Optional[int]:
        return self.slope_array_2d.shape[1]

    @property
    def is_affine(self) -> Optional[bool]:
        return True

    @property
    def is_strictly_convex(self) -> Optional[bool]:
        return False

    @property
    def is_convex(self) -> Optional[bool]:
        return True

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        logger.debug(x_array_2d.shape)
        logger.debug(self.slope_array_2d.shape)
        logger.debug(self.intercept_array_1d.shape)
        return x_array_2d.dot(self.slope_array_2d) + self.intercept_array_1d
