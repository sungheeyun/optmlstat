"""
linear predictor
"""

import numpy as np
from numpy import ndarray, hstack, ones

from optmlstat.functions.function_base import FunctionBase


class LinearPredictor(FunctionBase):
    """
    Linear predictor with basis functions.
    """

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
    def is_twice_differentiable(self) -> bool:
        raise NotImplementedError()

    def _hessian(self, x_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def is_strictly_concave(self) -> bool:
        raise NotImplementedError()

    @property
    def is_concave(self) -> bool:
        raise NotImplementedError()

    @property
    def is_differentiable(self) -> bool:
        raise NotImplementedError()

    def _jacobian(self, x_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def conjugate(self) -> FunctionBase:
        raise NotImplementedError()

    def conjugate_arg(self, z_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __init__(self, coef: ndarray, basis_function: FunctionBase) -> None:
        self.coef: ndarray = coef
        self.basis_function: FunctionBase = basis_function

        self._is_affine: bool = self.basis_function.is_affine
        self._is_convex: bool = self.basis_function.is_convex and bool((coef >= 0.0).all())
        self._is_strictly_convex: bool = self.basis_function.is_strictly_convex and bool(
            (coef > 0.0).all()
        )

    @property
    def num_inputs(self) -> int:
        return self.basis_function.num_inputs

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def is_affine(self) -> bool:
        return self._is_affine

    @property
    def is_strictly_convex(self) -> bool:
        return self._is_strictly_convex

    @property
    def is_convex(self) -> bool:
        return self._is_convex

    def _get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        z_array_2d: ndarray = hstack(
            (
                self.basis_function.get_y_values_2d(x_array_2d),
                ones((x_array_2d.shape[0], 1)),
            )
        )
        return z_array_2d.dot(self.coef)
