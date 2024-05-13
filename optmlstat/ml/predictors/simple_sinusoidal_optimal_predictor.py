"""
"""

import numpy as np
from numpy import ndarray, sin, pi

from optmlstat.functions.function_base import FunctionBase


class SimpleSinusoidalOptimalPredictor(FunctionBase):
    """
    An optimal predictor for SimpleSinusoidalSampler in least-square-mean sense.
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

    @property
    def num_inputs(self) -> int:
        raise NotImplementedError()

    @property
    def num_outputs(self) -> int:
        raise NotImplementedError()

    @property
    def is_affine(self) -> bool:
        return False

    @property
    def is_strictly_convex(self) -> bool:
        return False

    @property
    def is_convex(self) -> bool:
        return False

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        y_hat_array_2d = sin((2.0 * pi) * x_array_2d)

        return y_hat_array_2d
