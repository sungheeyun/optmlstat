"""
test numpy.linalg
"""

import unittest
from logging import Logger, getLogger

from numpy import ndarray, allclose, power, int32
from numpy.random import randn
from numpy.linalg import solve, lstsq, svd, norm
from freq_used.logging_utils import set_logging_basic_config


logger: Logger = getLogger()


class TestNumpyLinAlg(unittest.TestCase):
    dim_of_domain: int = 100
    dim_of_range: int = 1000
    num_of_data: int = 50

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_solve_1d(self) -> None:
        dim_of_domain: int = TestNumpyLinAlg.dim_of_domain

        a_array_2d: ndarray = randn(dim_of_domain, dim_of_domain)
        b_array_1d: ndarray = randn(dim_of_domain)

        x_array_1d: ndarray = solve(a_array_2d, b_array_1d)

        self.assertEqual(x_array_1d.ndim, 1)
        self.assertEqual(x_array_1d.shape, (dim_of_domain,))
        self.assertTrue(allclose(x_array_1d.dot(a_array_2d.T), b_array_1d))

    def test_solve_2d(self) -> None:
        dim_of_domain: int = TestNumpyLinAlg.dim_of_domain
        num_of_data: int = TestNumpyLinAlg.num_of_data

        a_array_2d: ndarray = randn(dim_of_domain, dim_of_domain)
        b_array_2d: ndarray = randn(dim_of_domain, num_of_data)

        x_array_2d: ndarray = solve(a_array_2d, b_array_2d)

        self.assertEqual(x_array_2d.ndim, 2)
        self.assertEqual(x_array_2d.shape, (dim_of_domain, num_of_data))
        self.assertTrue(allclose(a_array_2d.dot(x_array_2d), b_array_2d))

    def test_lstsq_1d(self) -> None:
        dim_of_domain: int = TestNumpyLinAlg.dim_of_domain
        dim_of_range: int = TestNumpyLinAlg.dim_of_range

        self.assertGreater(dim_of_range, dim_of_domain)

        a_array_2d: ndarray = randn(dim_of_range, dim_of_domain)
        b_array_1d: ndarray = randn(dim_of_range)

        x_array_1d, residuals_1d, rank, sv_array_1d = lstsq(a_array_2d, b_array_1d, rcond=None)

        self.assertTrue(isinstance(x_array_1d, ndarray))
        self.assertTrue(isinstance(residuals_1d, ndarray))
        self.assertTrue(isinstance(rank, int32))  # type:ignore
        self.assertTrue(isinstance(sv_array_1d, ndarray))

        self.assertEqual(x_array_1d.shape, (dim_of_domain,))
        self.assertEqual(residuals_1d.shape, (1,))
        self.assertEqual(rank, dim_of_domain)
        self.assertEqual(sv_array_1d.shape, (dim_of_domain,))

        self.assertTrue(allclose(sv_array_1d, svd(a_array_2d)[1]))

        self.assertEqual(a_array_2d.dot(x_array_1d).shape, b_array_1d.shape)
        self.assertTrue(
            allclose(
                norm(a_array_2d.dot(x_array_1d) - b_array_1d) ** 2,
                residuals_1d[0],
            )
        )

    def test_lstsq_2d(self) -> None:
        dim_of_domain: int = TestNumpyLinAlg.dim_of_domain
        dim_of_range: int = TestNumpyLinAlg.dim_of_range
        num_of_data: int = TestNumpyLinAlg.num_of_data

        self.assertGreater(dim_of_range, dim_of_domain)

        a_array_2d: ndarray = randn(dim_of_range, dim_of_domain)
        b_array_2d: ndarray = randn(dim_of_range, num_of_data)

        x_array_2d, residuals_1d, rank, sv_array_1d = lstsq(a_array_2d, b_array_2d, rcond=None)

        self.assertTrue(isinstance(x_array_2d, ndarray))
        self.assertTrue(isinstance(residuals_1d, ndarray))
        self.assertTrue(isinstance(rank, int32))  # type:ignore
        self.assertTrue(isinstance(sv_array_1d, ndarray))

        self.assertEqual(x_array_2d.shape, (dim_of_domain, num_of_data))
        self.assertEqual(residuals_1d.shape, (num_of_data,))
        self.assertEqual(rank, dim_of_domain)
        self.assertEqual(sv_array_1d.shape, (dim_of_domain,))

        self.assertTrue(allclose(sv_array_1d, svd(a_array_2d)[1]))

        self.assertEqual(a_array_2d.dot(x_array_2d).shape, b_array_2d.shape)
        self.assertTrue(
            allclose(
                power(a_array_2d.dot(x_array_2d) - b_array_2d, 2.0).sum(axis=0),
                residuals_1d,
            )
        )


if __name__ == "__main__":
    unittest.main()
