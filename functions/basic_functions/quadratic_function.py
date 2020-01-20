from typing import Optional
from logging import Logger, getLogger

from numpy import ndarray

from functions.function_base import FunctionBase


logger: Logger = getLogger()


class QuadraticFunction(FunctionBase):
    """
    Quadratic function.

    x^T A x + b^T x + c

    """

    def __init__(
        self, intercept_array_1d: ndarray, slope_array_2d: ndarray, quad_array_3d: Optional[ndarray] = None
    ) -> None:
        """
        If n is the number of inputs and m is that of outputs,

        Parameters
        ----------
        quad_array_3d:
         n-by-n-by-m ndarray
        slope_array_2d:
         n-by-m ndarray
        intercept_array_1d: m
         m ndarray
        """
        # check dimensionsj
        assert quad_array_3d is None or quad_array_3d.ndim == 3, quad_array_3d.ndim
        assert slope_array_2d.ndim == 2, slope_array_2d.ndim
        assert intercept_array_1d.ndim == 1, intercept_array_1d.ndim

        # check number of inputs
        assert quad_array_3d is None or quad_array_3d.shape[0] == slope_array_2d.shape[0]
        assert quad_array_3d is None or quad_array_3d.shape[1] == slope_array_2d.shape[0]

        # check number of outputs
        assert slope_array_2d.shape[1] == intercept_array_1d.size, (slope_array_2d.shape, intercept_array_1d.shape)
        assert quad_array_3d is None or quad_array_3d.shape[2] == intercept_array_1d.size, (
            slope_array_2d.shape,
            intercept_array_1d.shape,
        )

        self.intercept_array_1d: ndarray = intercept_array_1d.copy()
        self.slope_array_2d: ndarray = slope_array_2d.copy()
        self.quad_array_3d: ndarray = None if quad_array_3d is None else quad_array_3d.copy()

        super(QuadraticFunction, self).__init__(self.slope_array_2d.shape[0], self.slope_array_2d.shape[1])

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        logger.debug(x_array_2d.shape)
        logger.debug(self.slope_array_2d.shape)
        logger.debug(self.intercept_array_1d.shape)
        y_array_2d = x_array_2d.dot(self.slope_array_2d) + self.intercept_array_1d

        if self.quad_array_3d is not None:
            for idx in range(self.quad_array_3d.shape[2]):
                y_array_2d[:, idx] += (x_array_2d.dot(self.quad_array_3d[:, :, idx]) * x_array_2d).sum(axis=1)

        return y_array_2d
