"""
test feasible Newton's method for equality constrained optimization problems
"""

import unittest
from logging import Logger, getLogger

import numpy as np
from numpy import random as nr

from freq_used.logging_utils import set_logging_basic_config

from optmlstat.opt.optalgs.optalg_base import OptAlgBase
from optmlstat.opt.optalgs.feasible_newtons_method_for_linear_eq_const_prob import (
    FeasibleNewtonsMethodForLinearEqConstProb,
)
from optmlstat.opt.some_opt_probs import SomeSimpleOptProbs
from optmlstat.opt.constants import LineSearchMethod
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.opt_prob import OptProb
from optmlstat.plotting.opt_res_plotter import OptimizationResultPlotter


logger: Logger = getLogger()


class TestFeasibleEqConstrainedNewtonsMethod(unittest.TestCase):
    TRAJ: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_with_simple_example(self) -> None:
        num_pnts: int = 5
        self._test_eq_cnst_opt(
            "small example", *SomeSimpleOptProbs.simple_linear_program_two_vars(num_pnts)
        )

    def test_with_quad_obj(self) -> None:
        nr.seed(760104)
        self._test_eq_cnst_opt(
            "quad", *SomeSimpleOptProbs.random_eq_const_cvx_quad_prob(100, 10, 5)
        )

    def test_with_lse_obj(self) -> None:
        nr.seed(76010)
        self._test_eq_cnst_opt("lse", *SomeSimpleOptProbs.random_eq_const_lse_prob(100, 10, 5))

    def _test_eq_cnst_opt(self, title: str, opt_prob: OptProb, initial_x_2d: np.ndarray) -> None:

        optalg: OptAlgBase = FeasibleNewtonsMethodForLinearEqConstProb(
            LineSearchMethod.BackTrackingLineSearch
        )

        opt_res: OptResults = optalg.solve(
            opt_prob, OptParams(back_tracking_line_search_beta=0.9), True, initial_x_2d
        )

        if opt_res.x_diff_norm < np.inf:
            self.assertAlmostEqual(opt_res.x_diff_norm, 0.0, places=4)

        if opt_res.lambda_diff_norm < np.inf:
            self.assertAlmostEqual(opt_res.lambda_diff_norm, 0.0)

        if opt_res.nu_diff_norm < np.inf:
            self.assertAlmostEqual(opt_res.nu_diff_norm, 0.0)

        OptimizationResultPlotter.standard_plotting(opt_res, title, no_trajectory=not self.TRAJ)


if __name__ == "__main__":
    unittest.main()
