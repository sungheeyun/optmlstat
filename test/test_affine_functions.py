import unittest

from numpy import ndarray, array, eye, arange, allclose
from numpy.random import randn

from functions.basic_functions.affine_function import AffineFunction


class TestAffineFunctions(unittest.TestCase):
    def test_simple_affine_function(self):
        num_inputs: int = 10
        num_data: int = 100

        slope_array: ndarray = eye(num_inputs)
        intercept_array: ndarray = arange(num_inputs)

        x_array_2d: ndarray = randn(num_data, num_inputs)

        affine_function: AffineFunction = AffineFunction(slope_array, intercept_array)

        y_array_2d: ndarray = affine_function.get_y_values_2d(x_array_2d)
        true_y_array_2d: ndarray = x_array_2d + array([intercept_array]).repeat(x_array_2d.shape[0], axis=0)

        self.assertTrue(allclose(y_array_2d, true_y_array_2d))


if __name__ == "__main__":
    unittest.main()
