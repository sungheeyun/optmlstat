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

from functions.basic_functions.quadratic_function import QuadraticFunction
from functions.basic_functions.affine_function import AffineFunction
from functions.example_functions import get_sum_function, get_sum_of_square_function
from opt.opt_prob import OptimizationProblem
from opt.opt_res import OptimizationResult
from opt.cvxopt.admm.dual_ascend import DualAscend
from opt.opt_iterate import OptimizationIterate
from opt.special_solvers import strictly_convex_quadratic_with_linear_equality_constraints
from plotting.opt_res_plotter import OptimizationResultPlotter


logging
logger: Logger = getLogger()


def get_fcn_name(frame_info: FrameInfo) -> str:
    return frame_info[3]


class TestDualAscend(unittest.TestCase):
    domain_dim: int = 10
    num_data_points: int = 3
    num_eq_cnst: int = 2
    ABS_TOLERANCE_USED_FOR_COMPARE: float = 1e-1

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__, level=eval(f"logging.{os.environ.get('TEST_LOG_LEVEL', 'INFO')}"))

    @classmethod
    def tearDownClass(cls) -> None:
        plt.show()

    def test_dual_ascend_with_simple_example(self) -> None:
        seed(7601)
        figure: Figure = self._test_dual_ascend_with_quadratic_problem(
            TestDualAscend._get_simple_quad_problem(),
            num_data_points=TestDualAscend.num_data_points,
            frame_info=stack()[0],
        )
        figure.show()

    def test_dual_ascend_with_quad_prob_with_random_eq_cnsts(self) -> None:
        seed(760104)
        figure: Figure = self._test_dual_ascend_with_quadratic_problem(
            TestDualAscend._get_quad_problem_with_random_eq_cnsts(TestDualAscend.num_eq_cnst),
            num_data_points=TestDualAscend.num_data_points,
            frame_info=stack()[0],
        )
        figure.show()

    def _test_dual_ascend_with_quadratic_problem(
        self, opt_prob: OptimizationProblem, *, num_data_points: int = 1, frame_info: FrameInfo
    ) -> Figure:

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
            -eq_cnst_fcn.intercept_array_1d,
        )

        # solve by dual ascend

        initial_x_point_2d: ndarray = randn(num_data_points, opt_prob.domain_dim)
        initial_nu_point_2d: ndarray = randn(num_data_points, opt_prob.num_eq_cnst)

        learning_rate: float = 0.01
        dual_ascend: DualAscend = DualAscend(learning_rate)
        opt_res: OptimizationResult = dual_ascend.solve(
            opt_prob, initial_x_array_2d=initial_x_point_2d, initial_nu_array_2d=initial_nu_point_2d
        )

        final_iterate: OptimizationIterate = opt_res.final_iterate

        logger.debug(final_iterate)

        logger.info(f"true_opt_x: {opt_x_array_1d}")
        logger.info(f"true_opt_y: {opt_nu_array_1d}")

        logger.debug(f"x diff: {final_iterate.x_array_2d - opt_x_array_1d}")
        logger.info(f"max x diff: {abs(final_iterate.x_array_2d - opt_x_array_1d).max()}")
        logger.debug(f"nu diff: {final_iterate.nu_array_2d - opt_nu_array_1d}")
        logger.info(f"max nu diff: {abs(final_iterate.nu_array_2d - opt_nu_array_1d).max()}")

        self.assertTrue(
            allclose(final_iterate.x_array_2d, opt_x_array_1d, atol=TestDualAscend.ABS_TOLERANCE_USED_FOR_COMPARE)
        )
        self.assertTrue(
            allclose(final_iterate.nu_array_2d, opt_nu_array_1d, atol=TestDualAscend.ABS_TOLERANCE_USED_FOR_COMPARE)
        )

        axis1: Axes
        axis2: Axes

        figure: Figure = get_figure(2, 1)
        axis1, axis2 = figure.get_axes()
        OptimizationResultPlotter(opt_res).plot_primal_and_dual_objs(axis1, "-", gap_axis=axis2)
        figure.suptitle(get_fcn_name(frame_info), fontsize=15)

        return figure

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
