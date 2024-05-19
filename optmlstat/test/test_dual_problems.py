"""
test OptProb.dual_prob method
"""

import json
import unittest

import numpy as np

from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.functions.special_functions.empty_function import EmptyFunction
from optmlstat.opt.opt_prob import OptProb


class TestDualProblems(unittest.TestCase):
    def test_something(self):
        opt_prob_1: OptProb = OptProb(EmptyFunction(2, 1))
        opt_prob_2: OptProb = OptProb(EmptyFunction(2, 1), EmptyFunction(2, 3))
        opt_prob_3: OptProb = OptProb(EmptyFunction(2, 1), None, EmptyFunction(2, 3))
        opt_prob_4: OptProb = OptProb(None, EmptyFunction(4, 2), EmptyFunction(4, 3))
        opt_prob_5: OptProb = OptProb(
            QuadraticFunction(np.eye(2)[:, :, np.newaxis], np.zeros((2, 1)), np.zeros(1))
        )

        print(json.dumps(opt_prob_1.dual_problem.to_json_data(), indent=2))
        print(json.dumps(opt_prob_2.dual_problem.to_json_data(), indent=2))
        print(json.dumps(opt_prob_3.dual_problem.to_json_data(), indent=2))
        print(json.dumps(opt_prob_4.dual_problem.to_json_data(), indent=2))
        print(json.dumps(opt_prob_5.dual_problem.to_json_data(), indent=2))
        self.assertEqual(True, True)  # add assertion here


if __name__ == "__main__":
    unittest.main()
