import unittest
from logging import Logger, getLogger
import logging
import os
from inspect import FrameInfo, stack

from numpy import ndarray, abs, allclose
from numpy.random import randn, seed
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from freq_used.logging import set_logging_basic_config
from freq_used.plotting import get_figure

from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.functions.basic_functions.affine_function import AffineFunction
from optmlstat.functions.example_functions import get_sum_function, get_sum_of_square_function
from optmlstat.opt.opt_prob import OptimizationProblem
from optmlstat.opt.opt_res import OptimizationResult
from optmlstat.opt.cvxopt.admm.dual_ascend import DualAscend
from optmlstat.opt.opt_iterate import OptimizationIterate
from optmlstat.opt.special_solvers import strictly_convex_quadratic_with_linear_equality_constraints
from optmlstat.opt.opt_parameter import OptimizationParameter
from optmlstat.opt.learning_rate.vanishing_learning_rate_strategy import VanishingLearningRateStrategy
from optmlstat.plotting.opt_res_plotter import OptimizationResultPlotter
from optmlstat.plotting.multi_axes_animation import MultiAxesAnimation


logging
logger: Logger = getLogger()


def get_fcn_name(frame_info: FrameInfo) -> str:
    return frame_info[3]


class TestDualAscend(unittest.TestCase):
    domain_dim: int = 20
    num_data_points: int = 5
    num_eq_cnst: int = 2
    abs_tolerance_used_for_compare: float = 1e-6
    rel_tolerance_used_for_compare: float = 1e-6

    # opt_param: OptimizationParameter = OptimizationParameter(0.1077, 100)
    opt_param: OptimizationParameter = OptimizationParameter(VanishingLearningRateStrategy(10e-3, 1.0, 200), 1000)

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__, level=eval(f"logging.{os.environ.get('TEST_LOG_LEVEL', 'INFO')}"))

    @classmethod
    def tearDownClass(cls) -> None:
        plt.show()

    def test_dual_ascend_with_simple_example(self) -> None:
        seed(760104)
        self._test_dual_ascend_with_quadratic_problem(
            TestDualAscend._get_simple_quad_problem(),
            num_data_points=TestDualAscend.num_data_points,
            frame_info=stack()[0],
        )

    def test_dual_ascend_with_quad_prob_with_random_eq_cnsts(self) -> None:
        seed(760104)
        self._test_dual_ascend_with_quadratic_problem(
            TestDualAscend._get_quad_problem_with_random_eq_cnsts(TestDualAscend.num_eq_cnst),
            num_data_points=TestDualAscend.num_data_points,
            frame_info=stack()[0],
        )

    def _test_dual_ascend_with_quadratic_problem(
        self, opt_prob: OptimizationProblem, *, num_data_points: int = 1, frame_info: FrameInfo
    ) -> None:

        assert isinstance(opt_prob.obj_fcn, QuadraticFunction)
        assert isinstance(opt_prob.eq_cnst_fcn, AffineFunction)

        obj_fcn: QuadraticFunction = opt_prob.obj_fcn
        eq_cnst_fcn: AffineFunction = opt_prob.eq_cnst_fcn

        logger.debug(str(opt_prob))

        # calculate true solution

        opt_x_array_1d: ndarray
        opt_nu_array_1d: ndarray
        opt_x_array_1d, opt_nu_array_1d = strictly_convex_quadratic_with_linear_equality_constraints(
            obj_fcn.quad_array_3d[:, :, 0],
            obj_fcn.slope_array_2d[:, 0],
            eq_cnst_fcn.slope_array_2d.T,
            - eq_cnst_fcn.intercept_array_1d,
        )

        # solve by dual ascend

        initial_x_point_2d: ndarray = randn(num_data_points, opt_prob.domain_dim)
        initial_nu_point_2d: ndarray = randn(num_data_points, opt_prob.num_eq_cnst)

        dual_ascend: DualAscend = DualAscend()
        opt_res: OptimizationResult = dual_ascend.solve(
            opt_prob,
            TestDualAscend.opt_param,
            initial_x_array_2d=initial_x_point_2d,
            initial_nu_array_2d=initial_nu_point_2d,
        )

        final_iterate: OptimizationIterate = opt_res.final_iterate

        logger.debug(final_iterate)

        logger.info(f"true_opt_x: {opt_x_array_1d}")
        logger.info(f"true_opt_y: {opt_nu_array_1d}")

        logger.debug(f"x diff: {final_iterate.x_array_2d - opt_x_array_1d}")
        logger.info(f"max x diff: {abs(final_iterate.x_array_2d - opt_x_array_1d).max()}")
        logger.debug(f"nu diff: {final_iterate.nu_array_2d - opt_nu_array_1d}")
        logger.info(f"max nu diff: {abs(final_iterate.nu_array_2d - opt_nu_array_1d).max()}")

        axis1: Axes
        axis2: Axes

        optimization_result_plotter: OptimizationResultPlotter = OptimizationResultPlotter(opt_res)

        figure: Figure = get_figure(
            2, 1, axis_width=3.0, axis_height=2.5, top_margin=0.5, bottom_margin=0.5, vertical_padding=0.5
        )
        axis1, axis2 = figure.get_axes()
        optimization_result_plotter.plot_primal_and_dual_objs(axis1, "-", gap_axis=axis2)
        figure.suptitle(get_fcn_name(frame_info), fontsize=10)
        figure.show()

        multi_axes_animation: MultiAxesAnimation = optimization_result_plotter.animate_primal_sol()
        multi_axes_animation.figure.show()

        logger.info(f"MAX_ERR_1: {abs(final_iterate.x_array_2d - opt_x_array_1d).max()}")
        self.assertTrue(
            allclose(
                final_iterate.x_array_2d,
                opt_x_array_1d,
                atol=TestDualAscend.abs_tolerance_used_for_compare,
                rtol=TestDualAscend.rel_tolerance_used_for_compare,
            )
        )

        logger.info(f"MAX_ERR_2: {abs(final_iterate.nu_array_2d - opt_nu_array_1d).max()}")
        self.assertTrue(
            allclose(
                final_iterate.nu_array_2d,
                opt_nu_array_1d,
                atol=TestDualAscend.abs_tolerance_used_for_compare,
                rtol=TestDualAscend.rel_tolerance_used_for_compare,
            )
        )

    @classmethod
    def _get_simple_quad_problem(cls) -> OptimizationProblem:
        obj_fcn: QuadraticFunction = get_sum_of_square_function(cls.domain_dim)
        eq_cnst_fcn: AffineFunction = get_sum_function(cls.domain_dim, -1.0)
        return OptimizationProblem(obj_fcn, eq_cnst_fcn)

    @classmethod
    def _get_quad_problem_with_random_eq_cnsts(cls, num_eq_cnsts: int) -> OptimizationProblem:
        obj_fcn: QuadraticFunction = get_sum_of_square_function(cls.domain_dim)
        eq_cnst_fcn: AffineFunction = AffineFunction(randn(cls.domain_dim, num_eq_cnsts), randn(num_eq_cnsts))
        return OptimizationProblem(obj_fcn, eq_cnst_fcn)


if __name__ == "__main__":
    unittest.main()
