"""
test both feasible & infeasible Newton's method for equality constrained optimization problems
"""

import unittest
from logging import Logger, getLogger

import numpy as np
from numpy import random as nr
from freq_used.logging_utils import set_logging_basic_config

from optmlstat.opt.constants import LineSearchMethod
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.feasible_newtons_method_for_linear_eq_const_prob import (
    FeasibleNewtonsMethodForLinearEqConstProb,
)
from optmlstat.opt.optalgs.infeasible_newtons_method_for_linear_eq_const_prob import (
    InfeasibleNewtonsMethodForLinearEqConstProb,
)
from optmlstat.opt.optalgs.optalg_base import OptAlgBase
from optmlstat.opt.some_opt_probs import SomeSimpleOptProbs
from optmlstat.plotting.opt_res_plotter import OptimizationResultPlotter

logger: Logger = getLogger()


class TestEqConstrainedNewtonsMethods(unittest.TestCase):
    TRAJ: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_feasible_newton_with_simple_example(self) -> None:
        num_pnts: int = 5
        self._test_eq_cnst_opt(
            "feasible",
            "small example",
            *SomeSimpleOptProbs.simple_linear_program_two_vars(num_pnts),
        )

    def test_infeasible_newton_with_simple_example(self) -> None:
        num_pnts: int = 5
        self._test_eq_cnst_opt(
            "infeasible",
            "small example",
            *SomeSimpleOptProbs.simple_linear_program_two_vars(num_pnts),
        )

    def test_feasible_with_quad_obj(self) -> None:
        # nr.seed(760104)
        num_vars: int = 100
        num_eq_cnsts: int = 60
        num_members: int = 5
        self._test_eq_cnst_opt(
            "feasible",
            "quad",
            *SomeSimpleOptProbs.random_eq_const_cvx_quad_prob(num_vars, num_eq_cnsts, num_members),
        )

    def test_infeasible_with_quad_obj(self) -> None:
        # nr.seed(760104)
        num_vars: int = 100
        num_eq_cnsts: int = 60
        num_members: int = 5
        self._test_eq_cnst_opt(
            "infeasible",
            "quad",
            *SomeSimpleOptProbs.random_eq_const_cvx_quad_prob(num_vars, num_eq_cnsts, num_members),
        )

    def test_feasible_with_lse_obj(self) -> None:
        # nr.seed(76010)
        num_vars: int = 10
        num_eq_cnsts: int = 8
        num_members: int = 5
        self._test_eq_cnst_opt(
            "feasible",
            "lse",
            *SomeSimpleOptProbs.random_eq_const_lse_prob(num_vars, num_eq_cnsts, num_members),
        )

    def _test_infeasible_with_lse_obj(self) -> None:
        # nr.seed(76010)
        num_vars: int = 10
        num_eq_cnsts: int = 8
        num_members: int = 5
        self._test_eq_cnst_opt(
            "infeasible",
            "lse",
            *SomeSimpleOptProbs.random_eq_const_lse_prob(num_vars, num_eq_cnsts, num_members),
        )

    def _test_eq_cnst_opt(
        self, alg_type: str, title: str, opt_prob: OptProb, initial_x_2d: np.ndarray
    ) -> None:

        optalg: OptAlgBase
        if alg_type == "feasible":
            optalg = FeasibleNewtonsMethodForLinearEqConstProb(
                LineSearchMethod.BackTrackingLineSearch
            )
        elif alg_type == "infeasible":
            optalg = InfeasibleNewtonsMethodForLinearEqConstProb(
                LineSearchMethod.BackTrackingLineSearch
            )
            initial_x_2d = nr.randn(*initial_x_2d.shape)
        else:
            assert False, alg_type

        opt_res: OptResults = optalg.solve(
            opt_prob,
            OptParams(back_tracking_line_search_beta=0.9, tolerance_on_newton_dec=1e-2),
            True,
            initial_x_2d,
        )

        # if opt_res.x_diff_norm < np.inf:
        #     self.assertAlmostEqual(opt_res.x_diff_norm, 0.0, places=4)
        #
        # if opt_res.lambda_diff_norm < np.inf:
        #     self.assertAlmostEqual(opt_res.lambda_diff_norm, 0.0)
        #
        # if opt_res.nu_diff_norm < np.inf:
        #     self.assertAlmostEqual(opt_res.nu_diff_norm, 0.0)

        OptimizationResultPlotter.standard_plotting(
            opt_res,
            f"{alg_type.capitalize()} Newton's method - {title}: # opt vars: {opt_prob.dim_domain}"
            f", # eq cnst: {opt_prob.num_eq_cnst}",
            no_trajectory=not self.TRAJ,
        )


if __name__ == "__main__":
    unittest.main()
