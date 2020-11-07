import unittest
from logging import Logger, getLogger

from numpy import ndarray, array, newaxis, allclose, moveaxis
from numpy.random import randn
from freq_used.logging_utils import set_logging_basic_config

from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction


logger: Logger = getLogger()


class TestQuadraticFunctions(unittest.TestCase):
    num_test_points: int = 1000

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_quadratic_functions_1_dim_input(self):
        """
        test with one-dimensional quadratic function:

        2 x^2 + 5 x + 6
        3 x^2 - 3 x + 7

        Returns
        -------

        """
        c_0_array: ndarray = array([6, 7], float)
        c_1_array: ndarray = array([5, -3], float)
        c_2_array: ndarray = array([2, 3], float)

        num_inputs: int = 1
        num_outputs: int = c_0_array.size

        intercept_array_1d: ndarray[float] = c_0_array
        slope_array_2d: ndarray = c_1_array[newaxis, :]
        quad_array_3d: ndarray = ndarray((num_inputs, num_inputs, c_0_array.size), float)

        for idx in range(c_2_array.size):
            quad_array_3d[:, :, idx] = c_2_array[idx]

        # create a QuadraticFunction

        quadratic_function: QuadraticFunction = QuadraticFunction(quad_array_3d, slope_array_2d, intercept_array_1d)

        # check dimensions

        self.assertEqual(quadratic_function.num_inputs, num_inputs)
        self.assertEqual(quadratic_function.num_outputs, num_outputs)

        # check evaluated values

        x_array_1d: ndarray = randn(TestQuadraticFunctions.num_test_points)
        y_values_2d: ndarray = quadratic_function.get_y_values_2d_from_x_values_1d(x_array_1d)

        true_y_values_2d: ndarray = ndarray((x_array_1d.size, num_outputs))
        for idx, x_value in enumerate(x_array_1d):
            true_y_values_2d[idx, :] = c_0_array + c_1_array * x_value + c_2_array * x_value ** 2.0

        self.assertTrue(allclose(y_values_2d, true_y_values_2d))

    def test_general_quadratic_functions(self):

        num_inputs: int = 33
        num_outputs: int = 44

        intercept_array_1d: ndarray = randn(num_outputs)
        slope_array_2d: ndarray = randn(num_inputs, num_outputs)
        quad_array_3d: ndarray = randn(num_inputs, num_inputs, num_outputs)

        quadratic_function: QuadraticFunction = QuadraticFunction(quad_array_3d, slope_array_2d, intercept_array_1d)

        x_array_2d: ndarray = randn(TestQuadraticFunctions.num_test_points, num_inputs)
        y_values_2d: ndarray = quadratic_function.get_y_values_2d(x_array_2d)

        true_y_values_2d: ndarray = ndarray((x_array_2d.shape[0], num_outputs))
        for idx_data, x_array_1d in enumerate(x_array_2d):
            true_y_values_2d[idx_data, :] = intercept_array_1d
            true_y_values_2d[idx_data, :] += x_array_1d.dot(slope_array_2d)

            for idx, quad_array_2d in enumerate(moveaxis(quad_array_3d, -1, 0)):
                true_y_values_2d[idx_data, idx] += x_array_1d.dot(quad_array_2d).dot(array([x_array_1d]).T)

        logger.info(y_values_2d)
        logger.info(true_y_values_2d)

        self.assertTrue(allclose(y_values_2d, true_y_values_2d))


if __name__ == "__main__":
    unittest.main()
