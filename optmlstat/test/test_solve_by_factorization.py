import unittest

import numpy as np
import numpy.random as nr
import numpy.linalg as la

from optmlstat.ml.modeling.bayesian_least_squares_base import BayesianLeastSquaresBase


class TestSolveByFactorization(unittest.TestCase):
    def test_solve_by_cholesky_factorization(self):
        vec_size: int = 10

        a_array_2d: np.ndarray = nr.randn(vec_size, vec_size)
        a_array_2d = np.dot(a_array_2d, a_array_2d.T)

        y_array_1d: np.ndarray = nr.randn(vec_size)

        lower_tri: np.ndarray = la.cholesky(a_array_2d)

        cls = BayesianLeastSquaresBase
        x_array_1d: np.ndarray = cls.solve_linear_sys_using_lower_tri_from_chol_fac(
            lower_tri, y_array_1d
        )

        print(la.norm(np.dot(a_array_2d, x_array_1d) - y_array_1d))
        self.assertLess(la.norm(np.dot(a_array_2d, x_array_1d) - y_array_1d), 1e-10)


if __name__ == "__main__":
    unittest.main()
