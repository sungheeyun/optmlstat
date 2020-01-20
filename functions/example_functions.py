from numpy import eye, newaxis, ndarray, zeros, ones

from functions.basic_functions.quadratic_function import QuadraticFunction
from functions.basic_functions.affine_function import AffineFunction


def get_sum_of_square_function(num_inputs: int) -> QuadraticFunction:
    intercept_array_1d: ndarray = zeros(1)
    slope_array_2d: ndarray = zeros((num_inputs, 1))
    quad_array_3d: ndarray = eye(num_inputs)[:, :, newaxis]

    return QuadraticFunction(intercept_array_1d, slope_array_2d, quad_array_3d)


def get_sum_function(num_inputs: int) -> AffineFunction:
    slope_array_2d: ndarray = ones((num_inputs, 1))
    intercept_array_1d: ndarray = zeros(1)
    return AffineFunction(slope_array_2d, intercept_array_1d)
