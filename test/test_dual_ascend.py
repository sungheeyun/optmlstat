import unittest
from logging import Logger, getLogger
import logging
import json
import os

from numpy import block, ndarray, newaxis, zeros, abs, allclose
from numpy.random import randn, seed
from numpy.linalg import solve
from freq_used.logging import set_logging_basic_config

from functions.function_base import FunctionBase
from functions.basic_functions.quadratic_function import QuadraticFunction
from functions.basic_functions.affine_function import AffineFunction
from functions.example_functions import get_sum_function, get_sum_of_square_function
from opt.opt_prob.optimization_problem import OptimizationProblem
from opt.optimization_result import OptimizationResult
from opt.cvxopt.admm.dual_ascend import DualAscend
from formatting import ndarray_to_list


logging
logger: Logger = getLogger()


class TestDualAscend(unittest.TestCase):
    domain_dim: int = 10
    num_data_points: int = 5

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__, level=eval(f"logging.{os.environ.get('TEST_LOG_LEVEL', 'INFO')}"))

    def _test_dual_ascend_with_simple_example(self) -> None:
        self._test_dual_ascend(TestDualAscend._get_simple_quad_problem(), TestDualAscend.num_data_points)

    def test_dual_ascend_with_quad_prob_with_random_eq_cnsts(self) -> None:
        seed(760104)
        self._test_dual_ascend(TestDualAscend._get_quad_problem_with_random_eq_cnsts(2), TestDualAscend.num_data_points)

    def _test_dual_ascend(self, opt_prob: OptimizationProblem, num_data_points: int = 1) -> None:

        obj_fcn: FunctionBase = opt_prob.obj_fcn
        eq_cnst_fcn: FunctionBase = opt_prob.eq_cnst_fcn

        assert isinstance(obj_fcn, QuadraticFunction)
        assert isinstance(eq_cnst_fcn, AffineFunction)

        logger.info(json.dumps(opt_prob.to_json_data(), indent=2))

        p = eq_cnst_fcn.intercept_array_1d.size

        kkt_a_array: ndarray = block(
            [
                [2.0 * obj_fcn.quad_array_3d[:, :, 0], eq_cnst_fcn.slope_array_2d],
                [eq_cnst_fcn.slope_array_2d.T, zeros((p, p))],
            ]
        )
        kkt_b_array: ndarray = block([-obj_fcn.slope_array_2d[:, 0], -eq_cnst_fcn.intercept_array_1d])

        logger.debug(kkt_a_array.shape)
        logger.debug(kkt_b_array.shape)

        opt_sol: ndarray = solve(kkt_a_array, kkt_b_array)

        logger.debug(opt_sol)
        logger.debug(opt_sol.shape)
        logger.debug(opt_sol.__class__)

        opt_x: ndarray = opt_sol[: opt_prob.domain_dim]
        opt_y: ndarray = opt_sol[opt_prob.domain_dim:]

        initial_x_point_2d: ndarray = randn(num_data_points, opt_prob.domain_dim)
        initial_nu_point_2d: ndarray = randn(num_data_points, opt_prob.num_eq_cnst)

        learning_rate: float = 0.01
        dual_ascend: DualAscend = DualAscend(learning_rate)
        opt_res: OptimizationResult = dual_ascend.solve(
            opt_prob,
            initial_x_array_2d=initial_x_point_2d,
            initial_nu_array_2d=initial_nu_point_2d,
        )

        logger.info(
            json.dumps(opt_prob.to_json_data(opt_res.opt_x), indent=2, default=ndarray_to_list)
        )
        logger.debug(f"opt_nu: {opt_res.opt_nu}")

        logger.info(json.dumps(opt_prob.to_json_data(opt_x[newaxis, :]), indent=2, default=ndarray_to_list))
        logger.debug(f"true_opt_y: {opt_y}")

        logger.debug(f"x diff: {opt_res.opt_x - opt_x}")
        logger.info(f"max x diff: {abs(opt_res.opt_x - opt_x).max()}")
        logger.debug(f"nu diff: {opt_res.opt_nu - opt_y}")
        logger.info(f"max nu diff: {abs(opt_res.opt_nu - opt_y).max()}")

        self.assertTrue(allclose(opt_res.opt_x - opt_x, 0.0))
        self.assertTrue(allclose(opt_res.opt_nu - opt_y, 0.0))

    @classmethod
    def _get_simple_quad_problem(cls) -> OptimizationProblem:
        obj_fcn: QuadraticFunction = get_sum_of_square_function(cls.domain_dim)
        eq_cnst_fcn: AffineFunction = get_sum_function(cls.domain_dim, -1.0)
        return OptimizationProblem(obj_fcn, eq_cnst_fcn)

    @classmethod
    def _get_quad_problem_with_random_eq_cnsts(cls, num_eq_cnsts: int) -> OptimizationProblem:
        obj_fcn: QuadraticFunction = get_sum_of_square_function(cls.domain_dim)
        eq_cnst_fcn: AffineFunction = AffineFunction(randn(cls.domain_dim, num_eq_cnsts), randn(num_eq_cnsts))
        return OptimizationProblem(obj_fcn, eq_cnst_fcn)


if __name__ == "__main__":
    unittest.main()
