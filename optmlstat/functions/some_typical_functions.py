"""
some typical functions
"""

import numpy as np

from optmlstat.functions.basic_functions.affine_function import AffineFunction
from optmlstat.functions.basic_functions.log_sum_exp import LogSumExp
from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.functions.function_base import FunctionBase


def get_sum_of_square_function(num_inputs: int) -> QuadraticFunction:
    intercept_array_1d: np.ndarray = np.zeros(1)
    slope_array_2d: np.ndarray = np.zeros((num_inputs, 1))
    quad_array_3d: np.ndarray = np.eye(num_inputs)[:, :, np.newaxis]

    return QuadraticFunction(quad_array_3d, slope_array_2d, intercept_array_1d)


def get_sum_function(num_inputs: int, constant: float = 0.0) -> AffineFunction:
    slope_array_2d: np.ndarray = np.ones((num_inputs, 1))
    intercept_array_1d: np.ndarray = np.ones(1) * constant
    return AffineFunction(slope_array_2d, intercept_array_1d)


def get_cvxopt_book_for_grad_method() -> FunctionBase:
    return LogSumExp([[[1.0, 3.0], [1.0, -3.0], [-1.0, 0.0]]], -0.1 * np.ones((1, 3)))
