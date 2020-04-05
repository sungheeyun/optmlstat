import unittest
from logging import Logger, getLogger

from numpy import ndarray, power, allclose
from numpy.random import randn
from freq_used.logging import set_logging_basic_config

from optmlstat.functions.function_base import FunctionBase
from optmlstat.functions.example_functions import get_sum_of_square_function, get_sum_function


logger: Logger = getLogger()


class TestBasicFunctions(unittest.TestCase):
    num_inputs: int = 30
    num_data_points: int = 100
    x_array_2d: ndarray

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)
        cls.x_array_2d = randn(cls.num_data_points, cls.num_inputs)

    def test_sum_of_squares_function(self):
        y_array_1d: ndarray = TestBasicFunctions._get_y_array_1d(
            get_sum_of_square_function(TestBasicFunctions.num_inputs)
        )
        true_y_array_1d: ndarray = power(TestBasicFunctions.x_array_2d, 2.0).sum(axis=1)

        logger.info(y_array_1d.shape)
        logger.info(true_y_array_1d.shape)
        logger.info(allclose(y_array_1d, true_y_array_1d))

        self.assertTrue(allclose(y_array_1d, true_y_array_1d))

    def test_sum_function(self):
        y_array_1d: ndarray = TestBasicFunctions._get_y_array_1d(get_sum_function(TestBasicFunctions.num_inputs))
        true_y_array_1d: ndarray = power(TestBasicFunctions.x_array_2d, 1.0).sum(axis=1)

        logger.info(y_array_1d.shape)
        logger.info(true_y_array_1d.shape)
        logger.info(allclose(y_array_1d, true_y_array_1d))

        self.assertTrue(allclose(y_array_1d, true_y_array_1d))

    @classmethod
    def _get_y_array_1d(cls, function: FunctionBase) -> ndarray:
        return function.get_y_values_2d(cls.x_array_2d).ravel()


if __name__ == "__main__":
    unittest.main()
