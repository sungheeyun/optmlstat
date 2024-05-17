"""
test feasible Newton's method for equality constrained optimization problems
"""

import unittest

import numpy as np

from freq_used.logging_utils import set_logging_basic_config

from optmlstat.opt.optalgs.optalg_base import OptAlgBase
from optmlstat.opt.optalgs.linear_eq_constrained_feasible_newtons_method import (
    LinearEqConstrainedFeasibleNewtonsMethod,
)
from optmlstat.opt.some_simple_opt_probs import SomeSimpleOptProbs
from optmlstat.opt.constants import LineSearchMethod
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_res import OptResults


class TestFeasibleEqConstrainedNewtonsMethod(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_feasible_eq_constrained_newtons_method(self) -> None:

        num_pnts: int = 10

        optalg: OptAlgBase = LinearEqConstrainedFeasibleNewtonsMethod(
            LineSearchMethod.BackTrackingLineSearch
        )

        opt_prob, initial_x_array_2d = SomeSimpleOptProbs.get_simple_linear_program_two_vars(
            num_pnts
        )

        opt_res: OptResults = optalg.solve(
            opt_prob,
            OptParams(
                0.0,
                100,
                back_tracking_line_search_alpha=0.2,
                back_tracking_line_search_beta=0.5,
                tolerance_on_newton_dec=1e-2,
            ),
            True,
            initial_x_array_2d=initial_x_array_2d,
        )

        self.assertAlmostEqual(opt_res.best_obj_values.min(), 0.5)
        self.assertTrue(
            np.allclose(
                opt_res.iteration_iterate_list[1][-1].primal_prob_evaluation.x_array_2d, [0.5, 0.5]
            )
        )


if __name__ == "__main__":
    unittest.main()
