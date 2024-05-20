"""
test for gradient descent method
"""

import inspect
import logging  # noqa: F401
import os
import unittest
from logging import getLogger, Logger

import numpy as np
import numpy.random as nr
from freq_used.logging_utils import set_logging_basic_config

from optmlstat.functions.basic_functions.log_sum_exp import LogSumExp
from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.functions.exceptions import ValueUnknownException
from optmlstat.functions.function_base import FunctionBase
from optmlstat.linalg.utils import get_random_pos_def_array, get_random_orthogonal_array
from optmlstat.opt.constants import LineSearchMethod
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.grad_descent import GradDescent
from optmlstat.opt.optalgs.optalg_base import OptAlgBase
from optmlstat.opt.optalgs.unconstrained_newtons_method import UnconstrainedNewtonsMethod
from optmlstat.plotting.opt_res_plotter import OptimizationResultPlotter

logger: Logger = getLogger()


class TestUncDerivBasedOptAlgs(unittest.TestCase):
    SHOW_TRAJECTORY: bool = False
    RANDOM_SEED: int = 760104
    NUM_DATA_POINTS: int = 5
    NUM_VARS: int = 10
    NUM_TERMS: int = 300

    OPT_PARAM: OptParams = OptParams(
        max_num_outer_iterations=100,
        back_tracking_line_search_alpha=0.2,
        back_tracking_line_search_beta=0.9,
    )

    OBJFCN_PRJ_MAT_INIT_PNTS_MAP: dict[str, tuple[FunctionBase, np.ndarray, np.ndarray]] = dict()

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(
            __file__,
            level=eval(f"logging.{os.environ.get('TEST_LOG_LEVEL', 'INFO')}"),
        )
        nr.seed(cls.RANDOM_SEED)

        cls.OBJFCN_PRJ_MAT_INIT_PNTS_MAP["lse"] = (
            LogSumExp(
                [1e-1 * nr.randn(cls.NUM_TERMS, cls.NUM_VARS)], 1e-1 * nr.randn(1, cls.NUM_TERMS)
            ),
            get_random_orthogonal_array(cls.NUM_VARS)[:, :2],
            nr.randn(cls.NUM_DATA_POINTS, cls.NUM_VARS),
        )

        cls.OBJFCN_PRJ_MAT_INIT_PNTS_MAP["quad"] = (
            QuadraticFunction(
                get_random_pos_def_array(np.logspace(-1.0, 1.0, cls.NUM_VARS))[:, :, np.newaxis],
                nr.randn(cls.NUM_VARS)[:, np.newaxis],
                188.0 * np.ones(1),
            ),
            get_random_orthogonal_array(cls.NUM_VARS)[:, :2],
            nr.randn(cls.NUM_DATA_POINTS, cls.NUM_VARS),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # from matplotlib import pyplot as plt
        # plt.show()
        pass

    def test_grad_descent_quad(self) -> None:

        self._test_unc_deriv_based_optalg(
            inspect.currentframe().f_code.co_name,  # type:ignore
            GradDescent(LineSearchMethod.BackTrackingLineSearch),
            "quad",
            atol=1.0,
        )

    def test_grad_descent_lse(self) -> None:
        self._test_unc_deriv_based_optalg(
            inspect.currentframe().f_code.co_name,  # type:ignore
            GradDescent(LineSearchMethod.BackTrackingLineSearch),
            "lse",
            atol=1.0,
        )

    def test_newtons_method_quad(self) -> None:
        self._test_unc_deriv_based_optalg(
            inspect.currentframe().f_code.co_name,  # type:ignore
            UnconstrainedNewtonsMethod(LineSearchMethod.BackTrackingLineSearch),
            "quad",
        )

    def test_newtons_method_lse(self) -> None:
        self._test_unc_deriv_based_optalg(
            inspect.currentframe().f_code.co_name,  # type:ignore
            UnconstrainedNewtonsMethod(LineSearchMethod.BackTrackingLineSearch),
            "lse",
        )

    def _test_unc_deriv_based_optalg(
        self,
        calling_method_name: str,
        optalg: OptAlgBase,
        objfcn_name: str,
        /,
        *,
        atol: float = 1e-6,
    ) -> None:

        objfcn, proj_mat, init_x_2d = self.OBJFCN_PRJ_MAT_INIT_PNTS_MAP[objfcn_name]

        opt_prob: OptProb = OptProb(objfcn)
        opt_res: OptResults = optalg.solve(
            opt_prob,
            self.OPT_PARAM,
            True,
            init_x_2d,
        )

        OptimizationResultPlotter.standard_plotting(
            opt_res,
            calling_method_name,
            no_trajectory=not self.SHOW_TRAJECTORY,
            proj_mat_2d=proj_mat,
        )

        # logger.info(opt_prob.optimum_point)
        # logger.info(opt_res.final_iterate.x_2d.mean(axis=0) - opt_prob.optimum_point)
        try:
            self.assertTrue(
                np.allclose(
                    opt_res.final_iterate.x_array_2d.mean(axis=0), opt_prob.optimum_point, atol=atol
                )
            )
        except ValueUnknownException:
            pass

        try:
            self.assertTrue(np.allclose(opt_res.best_obj_values, opt_prob.optimum_value, atol=atol))
        except ValueUnknownException:
            pass


if __name__ == "__main__":
    unittest.main()
