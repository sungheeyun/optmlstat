from typing import Optional

import numpy as np
from numpy import ndarray

from optmlstat.functions.function_base import FunctionBase


class IdentityFunction(FunctionBase):
    def jacobian(self, x_array_2d: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError()

    def __init__(self):
        super().__init__()

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
