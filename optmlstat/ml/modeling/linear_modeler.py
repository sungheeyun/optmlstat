from typing import Optional
from copy import copy

from numpy import ndarray, hstack, ones, sqrt, eye, vstack, zeros
from numpy.linalg import lstsq

from functions.basic_functions.identity_function import IdentityFunction
from functions.function_base import FunctionBase
from ml.modeling.modeler_base import ModelerBase
from ml.modeling.modeling_result import ModelingResult
from ml.predictors.linear_predictor import LinearPredictor


class LinearModeler(ModelerBase):
    """
    Regularized linear least-square fitting using basis function.
    """

    def __init__(
        self,
        basis_function: Optional[FunctionBase] = None,
        reg_coef: float = 0.0,
    ):
        """
        :param basis_function:
        :param reg_coef:
         The coefficient for the 2-norm
        """
        self.basis_function: FunctionBase
        if basis_function is None:
            self.basis_function = IdentityFunction()
        else:
            self.basis_function = basis_function

        self.reg_coef: float = reg_coef

        self.coef: ndarray | None = None

    def train(self, x_array_2d: ndarray, y_array_2d: ndarray, **kwargs) -> ModelingResult:
        z_array_2d: ndarray = hstack(
            (
                self.basis_function.get_y_values_2d(x_array_2d),
                ones((x_array_2d.shape[0], 1)),
            )
        )

        a_array_2d: ndarray = vstack((z_array_2d, sqrt(self.reg_coef) * eye(z_array_2d.shape[1])))
        b_array_2d: ndarray = vstack(
            (y_array_2d, zeros((z_array_2d.shape[1], y_array_2d.shape[1])))
        )

        self.coef, residuals, rank, s = lstsq(a_array_2d, b_array_2d, rcond=None)

        return ModelingResult()

    def get_predictor(self) -> FunctionBase:
        if self.coef is None:
            raise Exception("Model has not been trained.")

        return LinearPredictor(self.coef.copy(), copy(self.basis_function))
