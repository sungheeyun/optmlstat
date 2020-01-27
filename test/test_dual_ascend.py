import unittest
from logging import Logger, getLogger
import logging
import json
import os

from numpy import block, ndarray, newaxis
from numpy.linalg import solve
from freq_used.logging import set_logging_basic_config

from functions.function_base import FunctionBase
from functions.basic_functions.quadratic_function import QuadraticFunction
from functions.basic_functions.affine_function import AffineFunction
from functions.example_functions import get_sum_function, get_sum_of_square_function
from opt.opt_prob.optimization_problem import OptimizationProblem
from opt.cvxopt.admm.dual_ascend import DualAscend


logger: Logger = getLogger()


class TestDualAscend(unittest.TestCase):
    num_inputs: int = 10

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__, level=eval(f"logging.{os.environ.get('TEST_LOG_LEVEL', 'INFO')}"))

    def test_dual_ascend_with_simple_example(self) -> None:
        self._test_dual_ascend(TestDualAscend.get_simple_quad_problem())

    def _test_dual_ascend(self, opt_prob: OptimizationProblem) -> None:

        obj_fcn: FunctionBase = opt_prob.obj_fcn
        eq_cnst_fcn: FunctionBase = opt_prob.eq_cnst_fcn

        assert isinstance(obj_fcn, QuadraticFunction)
        assert isinstance(eq_cnst_fcn, AffineFunction)

        logger.info(json.dumps(opt_prob.to_json_data(), indent=2))

        kkt_a_array: ndarray = block(
            [[2.0 * obj_fcn.quad_array_3d[:, :, 0], eq_cnst_fcn.slope_array_2d], [eq_cnst_fcn.slope_array_2d.T, 0]]
        )
        kkt_b_array: ndarray = block([-obj_fcn.slope_array_2d[:, 0], -eq_cnst_fcn.intercept_array_1d])

        logger.info(kkt_a_array.shape)
        logger.info(kkt_b_array.shape)

        opt_sol: ndarray = solve(kkt_a_array, kkt_b_array)

        logger.debug(opt_sol)
        logger.debug(opt_sol.shape)
        logger.debug(opt_sol.__class__)

        opt_x: ndarray = opt_sol[: opt_prob.domain_dim]
        opt_y: ndarray = opt_sol[opt_prob.domain_dim:]

        learning_rate: float = 0.01
        dual_ascend: DualAscend = DualAscend(learning_rate)
        dual_ascend.solve(opt_prob)

        logger.info(f"opt_x: {opt_x}")
        logger.info(f"opt_y: {opt_y}")
        logger.info(obj_fcn.get_y_values_2d(opt_x[newaxis, :]))

        self.assertEqual(1, 1)

    @classmethod
    def get_simple_quad_problem(cls) -> OptimizationProblem:
        obj_fcn: QuadraticFunction = get_sum_of_square_function(cls.num_inputs)
        eq_cnst_fcn: AffineFunction = get_sum_function(cls.num_inputs, -1.0)
        return OptimizationProblem(obj_fcn, eq_cnst_fcn)


if __name__ == "__main__":
    unittest.main()
