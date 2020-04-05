import unittest
from logging import Logger, getLogger

from numpy import eye, zeros, ndarray, newaxis, ones, allclose, array
from freq_used.logging import set_logging_basic_config

from optmlstat.opt.special_solvers import strictly_convex_quadratic_with_linear_equality_constraints

logger: Logger = getLogger()


class TestSpecialSolvers(unittest.TestCase):
    num_primary_vars: int = 10

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_strictly_convex_quadratic_with_linear_equality_constraints(self) -> None:
        for num_primary_vars in [10, 100, 1000]:
            self._test_strictly_convex_quadratic_with_linear_equality_constraints(num_primary_vars)

    def _test_strictly_convex_quadratic_with_linear_equality_constraints(self, num_primary_vars: int) -> None:
        p_array_2d: ndarray = eye(num_primary_vars)
        q_array_1d: ndarray = zeros(num_primary_vars)
        a_array_2d: ndarray = ones(num_primary_vars)[newaxis, :]
        b_array_1d: ndarray = ones(1)
        opt_x_1d, opt_nu_1d = strictly_convex_quadratic_with_linear_equality_constraints(
            p_array_2d, q_array_1d, a_array_2d, b_array_1d
        )

        logger.debug(f"opt_x_1d: {opt_x_1d}")
        logger.info(f"opt_nu_1d: {opt_nu_1d}")

        self.assertTrue(allclose(opt_x_1d, 1.0 / num_primary_vars))

    def test_simple_linear_equality_constraints(self) -> None:
        """
        minimize x^2 + y^2
        subject to 2x + y = 1

        The optimal solution is x = 2/5 and y = 1/5.
        """
        p_array_2d: ndarray = eye(2)
        q_array_1d: ndarray = zeros(2)
        a_array_2d: ndarray = array([2, 1], float)[newaxis, :]
        b_array_1d: ndarray = ones(1)

        opt_x_1d, opt_nu_1d = strictly_convex_quadratic_with_linear_equality_constraints(
            p_array_2d, q_array_1d, a_array_2d, b_array_1d
        )

        logger.info(f"opt_x_1d: {opt_x_1d}")
        logger.info(f"opt_nu_1d: {opt_nu_1d}")

        self.assertTrue(allclose(opt_x_1d, [0.4, 0.2]))


if __name__ == "__main__":
    unittest.main()
