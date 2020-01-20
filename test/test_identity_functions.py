import unittest

from numpy import ndarray, allclose
from numpy.random import randn

from functions.basic_functions.identity_function import IdentityFunction


class TestIdentityFunction(unittest.TestCase):
    def test_identity_function(self):
        num_inputs: int = 100
        num_data: int = 1000

        identity_function: IdentityFunction = IdentityFunction()

        x_array_2d: ndarray = randn(num_data, num_inputs)
        y_array_2d: ndarray = identity_function.get_y_values_2d(x_array_2d)

        self.assertTrue(allclose(y_array_2d, x_array_2d))

    def test_num_dimensions(self):
        identity_function: IdentityFunction = IdentityFunction()

        self.assertIsNone(identity_function.num_inputs)
        self.assertIsNone(identity_function.num_outputs)


if __name__ == "__main__":
    unittest.main()
