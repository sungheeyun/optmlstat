from numpy import ndarray, hstack, ones

from optmlstat.functions.basis_functions.basis_function_base import BasisFunctionBase
from optmlstat.ml.predictors.predictor_base import PredictorBase


class LinearPredictor(PredictorBase):
    """
    Linear predictor with basis functions.
    """
    def __init__(self, coef: ndarray, basis_function: BasisFunctionBase) -> None:
        self.coef: ndarray = coef
        self.basis_function: BasisFunctionBase = basis_function

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        z_array_2d: ndarray = hstack((self.basis_function.get_y_values_2d(x_array_2d), ones((x_array_2d.shape[0], 1))))
        return z_array_2d.dot(self.coef)
