"""
test for gradient descent method
"""

import logging  # noqa: F401
import os
import unittest
from logging import getLogger, Logger

import numpy as np
import numpy.random as nr
from freq_used.logging_utils import set_logging_basic_config
from freq_used.plotting import get_figure
from matplotlib.figure import Figure

from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.functions.function_base import FunctionBase
from optmlstat.opt.constants import LineSearchMethod
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.grad_descent import GradDescent
from optmlstat.opt.optalgs.optalg_base import OptAlgBase
from optmlstat.opt.optalgs.unconstrained_newtons_method import UnconstrainedNewtonsMethod
from optmlstat.plotting.opt_res_plotter import OptimizationResultPlotter

logger: Logger = getLogger()


class TestGradDescent(unittest.TestCase):
    RANDOM_SEED: int = 760104
    NUM_DATA_POINTS: int = 10
    # abs_tolerance_used_for_compare: float = 1e-6
    # rel_tolerance_used_for_compare: float = 1e-6

    # opt_param: OptimizationParameter = OptimizationParameter(0.1077, 100)
    opt_param: OptParams = OptParams(
        0.1,
        100,
        back_tracking_line_search_alpha=0.25,
        back_tracking_line_search_beta=0.5,
        tolerance_on_grad=1e-6,
    )

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(
            __file__,
            level=eval(f"logging.{os.environ.get('TEST_LOG_LEVEL', 'INFO')}"),
        )
        nr.seed(cls.RANDOM_SEED)

    @classmethod
    def tearDownClass(cls) -> None:
        from matplotlib import pyplot as plt

        plt.show()
        pass

    def test_grad_descent(self) -> None:
        """
        test gradient method
        """
        self._test_unc_deriv_based_optalg(GradDescent(LineSearchMethod.BackTrackingLineSearch))

    def _test_newtons_method(self) -> None:
        """
        test Newton's method
        """
        self._test_unc_deriv_based_optalg(
            UnconstrainedNewtonsMethod(LineSearchMethod.BackTrackingLineSearch)
        )

    def _test_unc_deriv_based_optalg(self, optalg: OptAlgBase) -> None:
        num_vars: int = 2

        obj_fcn: FunctionBase = QuadraticFunction(
            np.diag([9, 1])[:, :, None], -2.0 * np.array([3.0, 1.0])[:, np.newaxis], np.zeros(1)
        )
        initial_x_2d: np.ndarray = 8 * nr.rand(self.NUM_DATA_POINTS, num_vars) - 4

        opt_prob: OptProb = OptProb(obj_fcn)

        opt_res: OptResults = optalg.solve(
            opt_prob, self.opt_param, True, initial_x_array_2d=initial_x_2d
        )
        opt_res.result_analysis()
        logger.info(opt_res.final_iterate.x_array_2d.mean(axis=0) - opt_prob.optimum_point)
        self.assertTrue(
            np.allclose(
                opt_res.final_iterate.x_array_2d.mean(axis=0), opt_prob.optimum_point, atol=1e-4
            )
        )
        self.assertTrue(np.allclose(opt_res.best_obj_values, opt_prob.optimum_value, atol=1e-8))

        figure: Figure = get_figure(
            2,
            2,
            axis_width=3.0,
            axis_height=2.5,
            top_margin=0.5,
            bottom_margin=0.5,
            vertical_padding=1.0,
        )
        ax1, ax2, ax3, ax4 = figure.get_axes()

        optimization_result_plotter: OptimizationResultPlotter = OptimizationResultPlotter(opt_res)
        optimization_result_plotter.plot_primal_and_dual_objs(ax1, ax3, ax4, ".-")
        optimization_result_plotter.animate_primal_sol(ax2, [ax1, ax3], interval=100.0)


if __name__ == "__main__":
    unittest.main()
