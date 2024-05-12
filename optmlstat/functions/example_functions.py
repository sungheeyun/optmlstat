from numpy import eye, newaxis, ndarray, zeros, ones

from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.functions.basic_functions.affine_function import AffineFunction
from optmlstat.functions.function_base import FunctionBase


def get_sum_of_square_function(num_inputs: int) -> QuadraticFunction:
    intercept_array_1d: ndarray = zeros(1)
    slope_array_2d: ndarray = zeros((num_inputs, 1))
    quad_array_3d: ndarray = eye(num_inputs)[:, :, newaxis]

    return QuadraticFunction(quad_array_3d, slope_array_2d, intercept_array_1d)


def get_sum_function(num_inputs: int, constant: float = 0.0) -> AffineFunction:
    slope_array_2d: ndarray = ones((num_inputs, 1))
    intercept_array_1d: ndarray = ones(1) * constant
    return AffineFunction(slope_array_2d, intercept_array_1d)


def get_cvxopt_book_for_grad_method() -> FunctionBase:
    pass
