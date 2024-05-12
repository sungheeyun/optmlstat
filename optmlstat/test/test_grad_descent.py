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
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.functions.function_base import FunctionBase
from optmlstat.opt.constants import LineSearchMethod
from optmlstat.opt.opt_iterate import OptimizationIterate
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.grad_descent import GradDescent
from optmlstat.plotting.opt_res_plotter import OptimizationResultPlotter

logger: Logger = getLogger()


class TestGradDescent(unittest.TestCase):
    RANDOM_SEED: int = 76010
    NUM_DATA_POINTS: int = 10
    # abs_tolerance_used_for_compare: float = 1e-6
    # rel_tolerance_used_for_compare: float = 1e-6

    # opt_param: OptimizationParameter = OptimizationParameter(0.1077, 100)
    opt_param: OptParams = OptParams(
        0.1,
        20,
        back_tracking_line_search_alpha=0.25,
        back_tracking_line_search_beta=0.5,
    )

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(
            __file__,
            level=eval(f"logging.{os.environ.get('TEST_LOG_LEVEL', 'INFO')}"),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        plt.show()

    def test_grad_descent(self) -> None:
        nr.seed(self.RANDOM_SEED)
        num_vars: int = 2

        obj_fcn: FunctionBase = QuadraticFunction(
            np.diag([30, 1])[:, :, None], 3 * np.ones((2, 1)), np.zeros(1)
        )
        initial_x_2d: np.ndarray = 8 * nr.rand(self.NUM_DATA_POINTS, num_vars) - 4
        # initial_x_2d: np.ndarray = np.array([[4.0, -4.0]])

        opt_prob: OptProb = OptProb(obj_fcn)

        grad_descent: GradDescent = GradDescent(LineSearchMethod.BackTrackingLineSearch)
        opt_res: OptResults = grad_descent.solve(opt_prob, self.opt_param, initial_x_2d)

        final_iterate: OptimizationIterate = opt_res.final_iterate
        logger.info(final_iterate.x_array_2d)

        figure: Figure = get_figure(
            2,
            1,
            axis_width=3.0,
            axis_height=2.5,
            top_margin=0.5,
            bottom_margin=0.5,
            vertical_padding=0.5,
        )
        axis1, axis2 = figure.get_axes()

        optimization_result_plotter: OptimizationResultPlotter = (
            OptimizationResultPlotter(opt_res)
        )
        optimization_result_plotter.plot_primal_and_dual_objs(
            axis1, "-", gap_axis=axis2
        )
        optimization_result_plotter.animate_primal_sol(interval=1000.0)


if __name__ == "__main__":
    unittest.main()
