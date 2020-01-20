from typing import Optional

from numpy import ndarray

from functions.function_base import FunctionBase


class IdentityFunction(FunctionBase):
    def __init__(self):
        super(IdentityFunction, self).__init__()

    @property
    def num_inputs(self) -> Optional[int]:
        return None

    @property
    def num_outputs(self) -> Optional[int]:
        return None

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
        return x_array_2d.copy()
