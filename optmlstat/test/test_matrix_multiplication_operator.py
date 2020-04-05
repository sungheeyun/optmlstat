import unittest
from logging import Logger, getLogger

from numpy import ndarray, array, arange, allclose
from freq_used.logging import set_logging_basic_config

from optmlstat.linalg.basic_operators.matrix_multiplication_operator import MatrixMultiplicationOperator

logger: Logger = getLogger()


class TestMatrixMultiplicationOperator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_matrix_multiplication_operator(self):

        array_2d: ndarray = array(arange(4), float).reshape((2, 2))
        array_1d: ndarray = array([1, 10], float)

        logger.info(array_2d)

        matrix_multiplication_operator: MatrixMultiplicationOperator = MatrixMultiplicationOperator(array_2d)

        logger.info(allclose(matrix_multiplication_operator.transform(array_1d), [10, 32]))

        self.assertTrue(allclose(matrix_multiplication_operator.transform(array_1d), [10, 32]))


if __name__ == '__main__':
    unittest.main()
