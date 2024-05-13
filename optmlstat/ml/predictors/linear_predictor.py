from typing import Optional

from numpy import ndarray, hstack, ones

from functions.function_base import FunctionBase


class LinearPredictor(FunctionBase):
    """
    Linear predictor with basis functions.
    """

    def __init__(self, coef: ndarray, basis_function: FunctionBase) -> None:
        self.coef: ndarray = coef
        self.basis_function: FunctionBase = basis_function

        self._is_affine: Optional[bool] = True if self.basis_function.is_affine else None
        self._is_convex: Optional[bool] = (
            True if self.basis_function.is_convex and (coef >= 0.0).all() else None
        )
        self._is_strictly_convex: Optional[bool] = (
            True if self.basis_function.is_strictly_convex and (coef > 0.0).all() else None
        )

    @property
    def num_inputs(self) -> Optional[int]:
        return self.basis_function.num_inputs

    @property
    def num_outputs(self) -> Optional[int]:
        return 1

    @property
    def is_affine(self) -> Optional[bool]:
        return self._is_affine

    @property
    def is_strictly_convex(self) -> Optional[bool]:
        return self._is_strictly_convex

    @property
    def is_convex(self) -> bool:
        return self._is_convex

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        z_array_2d: ndarray = hstack(
            (
                self.basis_function.get_y_values_2d(x_array_2d),
                ones((x_array_2d.shape[0], 1)),
            )
        )
        return z_array_2d.dot(self.coef)
