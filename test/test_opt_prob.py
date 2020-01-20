import unittest
from logging import Logger, getLogger
import json

from freq_used.logging import set_logging_basic_config

from functions.example_functions import get_sum_of_square_function, get_sum_function
from functions.basic_functions.quadratic_function import QuadraticFunction
from functions.basic_functions.affine_function import AffineFunction
from opt.opt_prob.optimization_problem import OptimizationProblem


logger: Logger = getLogger()


class TestOptimizationProblem(unittest.TestCase):
    num_inputs: int = 3

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_something(self):
        obj_fcn: QuadraticFunction = get_sum_of_square_function(TestOptimizationProblem.num_inputs)
        eq_cnst_fcn: AffineFunction = get_sum_function(TestOptimizationProblem.num_inputs, -1.0)

        optimization_problem: OptimizationProblem = OptimizationProblem(obj_fcn, eq_cnst_fcn)

        logger.info("optimization_problem:")
        logger.info(f"{json.dumps(optimization_problem.to_json_data(), indent=2)}")

        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
