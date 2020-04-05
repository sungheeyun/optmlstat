import unittest
from logging import Logger, getLogger

from numpy import ndarray, allclose
from numpy.random import randn
from freq_used.logging import set_logging_basic_config


logger: Logger = getLogger()


class TestNumpyBasicOperations(unittest.TestCase):
    dim_of_domain: int = 100
    dim_of_range: int = 1000
    num_of_data: int = 50

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_solve_1d(self) -> None:
        dim_of_domain: int = TestNumpyBasicOperations.dim_of_domain
        num_of_data: int = TestNumpyBasicOperations.num_of_data

        array_2d: ndarray = randn(num_of_data, dim_of_domain)
        array_1d: ndarray = randn(dim_of_domain)

        sum_array_2d: ndarray = array_2d + array_1d
        true_sum_array_2d: ndarray = ndarray(array_2d.shape, dtype=float)

        for idx, row_array_1d in enumerate(array_2d):
            true_sum_array_2d[idx, :] = row_array_1d + array_1d

        self.assertTrue(allclose(sum_array_2d, true_sum_array_2d))


if __name__ == "__main__":
    unittest.main()
