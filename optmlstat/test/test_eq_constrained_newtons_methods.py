"""
test both feasible & infeasible Newton's method for equality constrained optimization problems
"""

import unittest
from logging import Logger, getLogger

import numpy as np
from numpy import random as nr
from freq_used.logging_utils import set_logging_basic_config
from scipy import linalg

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

    OPT_PROB_DICT: dict[str, tuple[OptProb, np.ndarray, np.ndarray]] = dict()
    final_sol: dict[str, np.ndarray] = dict()

    quad_num_vars: int = 200
    quad_num_eq_cnsts: int = 150
    quad_num_members: int = 5
    small_num_members: int = 5
    lse_num_vars: int = 10
    lse_num_eq_cnsts: int = 8
    lse_num_members: int = 5

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)
        nr.seed(760104)

        cls.OPT_PROB_DICT["quad"] = SomeSimpleOptProbs.random_eq_const_cvx_quad_prob(
            cls.quad_num_vars, cls.quad_num_eq_cnsts, cls.quad_num_members
        )

        cls.OPT_PROB_DICT["small example"] = SomeSimpleOptProbs.simple_linear_program_two_vars(
            cls.small_num_members
        )

        cls.OPT_PROB_DICT["lse"] = SomeSimpleOptProbs.random_eq_const_lse_prob(
            cls.lse_num_vars, cls.lse_num_eq_cnsts, cls.lse_num_members
        )

    def test_feasible_newton_with_simple_example(self) -> None:
        self._test_eq_cnst_opt("feasible", "small example")

    def test_infeasible_newton_with_simple_example(self) -> None:
        self._test_eq_cnst_opt("infeasible", "small example")

    def test_feasible_with_quad_obj(self) -> None:
        self._test_eq_cnst_opt("feasible", "quad")

    def test_infeasible_with_quad_obj(self) -> None:
        self._test_eq_cnst_opt("infeasible", "quad")

    def test_feasible_with_lse_obj(self) -> None:
        self._test_eq_cnst_opt("feasible", "lse")

    def test_infeasible_with_lse_obj(self) -> None:
        self._test_eq_cnst_opt("infeasible", "lse")

    def _test_eq_cnst_opt(self, alg_type: str, prob_name: str) -> None:

        opt_prob, initial_x_2d, proj_2d = self.OPT_PROB_DICT[prob_name]

        optalg: OptAlgBase
        if alg_type == "feasible":
            optalg = FeasibleNewtonsMethodForLinearEqConstProb(
                LineSearchMethod.BackTrackingLineSearch
            )
        elif alg_type == "infeasible":
            optalg = InfeasibleNewtonsMethodForLinearEqConstProb(
                LineSearchMethod.BackTrackingLineSearch
            )
            initial_x_2d = 0.1 * nr.randn(*initial_x_2d.shape)
        else:
            assert False, alg_type

        opt_res: OptResults = optalg.solve(
            opt_prob,
            OptParams(),
            True,
            initial_x_2d,
        )

        if opt_res.x_diff_norm < np.inf:
            self.assertAlmostEqual(opt_res.x_diff_norm, 0.0)
            # print(opt_res.x_diff_norm)

        if opt_res.lambda_diff_norm < np.inf:
            self.assertAlmostEqual(opt_res.lambda_diff_norm, 0.0)
            # print(opt_res.lambda_diff_norm)

        if opt_res.nu_diff_norm < np.inf:
            self.assertAlmostEqual(opt_res.nu_diff_norm, 0.0)
            # print(opt_res.nu_diff_norm)

        if prob_name in self.final_sol:
            self.assertLess(linalg.norm((opt_res.final_x_2d - self.final_sol[prob_name])), 1e-3)
        else:
            self.final_sol[prob_name] = opt_res.final_x_2d.copy()

        OptimizationResultPlotter.standard_plotting(
            opt_res,
            f"{alg_type.capitalize()} Newton's method - {prob_name}:"
            f" # opt vars: {opt_prob.dim_domain}, # eq cnst: {opt_prob.num_eq_cnst}",
            no_trajectory=not self.TRAJ,
            proj_mat_2d=proj_2d,
        )


if __name__ == "__main__":
    unittest.main()
