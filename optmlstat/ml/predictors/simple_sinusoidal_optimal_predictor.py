from typing import Optional

from numpy import ndarray, sin, pi

from functions.function_base import FunctionBase


class SimpleSinusoidalOptimalPredictor(FunctionBase):
    """
    An optimal predictor for SimpleSinusoidalSampler in least-square-mean sense.
    """

    @property
    def num_inputs(self) -> Optional[int]:
        return None

    @property
    def num_outputs(self) -> Optional[int]:
        return None

    @property
    def is_affine(self) -> Optional[bool]:
        return False

    @property
    def is_strictly_convex(self) -> Optional[bool]:
        return False

    @property
    def is_convex(self) -> Optional[bool]:
        return False

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        y_hat_array_2d = sin((2.0 * pi) * x_array_2d)

        return y_hat_array_2d
