"""

"""

import logging
import os
import unittest
from inspect import FrameInfo, stack
from logging import Logger, getLogger

import matplotlib as mpl
from freq_used.logging_utils import set_logging_basic_config
from freq_used.plotting import get_figure
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray, abs, allclose
from numpy.random import randn, seed

from optmlstat.functions.basic_functions.affine_function import AffineFunction
from optmlstat.functions.basic_functions.quadratic_function import (
    QuadraticFunction,
)
from optmlstat.functions.some_typical_functions import (
    get_sum_function,
    get_sum_of_square_function,
)
from optmlstat.opt.cvxopt.admm.dual_ascend import DualAscend
from optmlstat.opt.learning_rate.vanishing_learning_rate_strategy import (
    VanishingLearningRateStrategy,
)
from optmlstat.opt.opt_iterate import OptimizationIterate
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.special_solvers import (
    strictly_convex_quadratic_with_linear_equality_constraints,
)
from optmlstat.plotting.opt_res_plotter import OptimizationResultPlotter

mpl.use("TkAgg")
logging
logger: Logger = getLogger()


def get_fcn_name(frame_info: FrameInfo) -> str:
    return frame_info[3]


class TestDualAscend(unittest.TestCase):
    FIXED_SEED: bool = True
    domain_dim: int = 20
    num_data_points: int = 5
    num_eq_cnst: int = 2
    abs_tolerance_used_for_compare: float = 1e-5
    rel_tolerance_used_for_compare: float = 1e-5

    # opt_param: OptimizationParameter = OptimizationParameter(0.1077, 100)
    opt_param: OptParams = OptParams(VanishingLearningRateStrategy(10e-3, 1.0, 200), 500)

    @classmethod
    def setUpClass(cls) -> None:
        assert os.environ.get("TEST_LOG_LEVEL", "INFO")
        set_logging_basic_config(
            __file__,
            level=eval(f"logging.{os.environ.get('TEST_LOG_LEVEL', 'INFO')}"),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # from matplotlib import pyplot as plt
        #
        # plt.show()
        pass

    def test_dual_ascend_with_simple_example(self) -> None:
        if self.FIXED_SEED:
            seed(760104)
        self._test_dual_ascend_with_quadratic_problem(
            TestDualAscend._get_simple_quad_problem(),
            num_data_points=TestDualAscend.num_data_points,
            frame_info=stack()[0],
        )

    def test_dual_ascend_with_quad_prob_with_random_eq_cnsts(self) -> None:
        if self.FIXED_SEED:
            seed(760104)
        self._test_dual_ascend_with_quadratic_problem(
            TestDualAscend._get_quad_problem_with_random_eq_cnsts(TestDualAscend.num_eq_cnst),
            num_data_points=TestDualAscend.num_data_points,
            frame_info=stack()[0],
        )

    def _test_dual_ascend_with_quadratic_problem(
        self,
        opt_prob: OptProb,
        *,
        num_data_points: int = 1,
        frame_info: FrameInfo,
    ) -> None:

        assert isinstance(opt_prob.obj_fcn, QuadraticFunction)
        assert isinstance(opt_prob.eq_cnst_fcn, AffineFunction)

        obj_fcn: QuadraticFunction = opt_prob.obj_fcn
        eq_cnst_fcn: AffineFunction = opt_prob.eq_cnst_fcn

        logger.debug(str(opt_prob))

        # calculate true solution

        opt_x_array_1d: ndarray
        opt_nu_array_1d: ndarray

        assert obj_fcn.quad_3d is not None

        (
            opt_x_array_1d,
            opt_nu_array_1d,
        ) = strictly_convex_quadratic_with_linear_equality_constraints(
            obj_fcn.quad_3d[:, :, 0],
            obj_fcn.slope_2d[:, 0],
            eq_cnst_fcn.slope_array_2d.T,
            -eq_cnst_fcn.intercept_array_1d,
        )

        # solve by dual ascend

        initial_x_point_2d: ndarray = randn(num_data_points, opt_prob.dim_domain)
        initial_nu_point_2d: ndarray = randn(num_data_points, opt_prob.num_eq_cnst)

        dual_ascend: DualAscend = DualAscend()
        opt_res: OptResults = dual_ascend.solve(
            opt_prob,
            TestDualAscend.opt_param,
            False,
            initial_x_point_2d,
            initial_nu_2d=initial_nu_point_2d,
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
            2,
            1,
            axis_width=3.0,
            axis_height=2.5,
            top_margin=0.5,
            bottom_margin=0.5,
            vertical_padding=0.5,
        )
        axis1, axis2 = figure.get_axes()
        optimization_result_plotter.plot_primal_and_dual_objs(axis1, axis2, axis2, "-")
        figure.suptitle(get_fcn_name(frame_info), fontsize=10)

        # optimization_result_plotter.animate_primal_sol()

        logger.info(f"MAX_ERR_1: {abs(final_iterate.x_array_2d - opt_x_array_1d).max()}")
        assert final_iterate.x_array_2d is not None
        # TODO (L) written on 17-May-2024
        #  need to figure out why below test does not pass right after i implemented
        #  feasible Newton's method for linearly equality constrained minimization
        #  i will need to look at this when i revisit dual ascend,
        #  or maybe never need to do it if i skip this and go to ADMM directly

        logger.debug(final_iterate.x_array_2d)
        logger.debug(opt_x_array_1d)
        logger.debug(final_iterate.x_array_2d - opt_x_array_1d)
        logger.debug(allclose(final_iterate.x_array_2d, opt_x_array_1d))
        logger.debug(TestDualAscend.abs_tolerance_used_for_compare)
        logger.debug(TestDualAscend.rel_tolerance_used_for_compare)
        logger.debug(
            allclose(
                final_iterate.x_array_2d,
                opt_x_array_1d,
                atol=TestDualAscend.abs_tolerance_used_for_compare,
                rtol=TestDualAscend.rel_tolerance_used_for_compare,
            )
        )
        self.assertTrue(
            allclose(
                final_iterate.x_array_2d,
                opt_x_array_1d,
                atol=TestDualAscend.abs_tolerance_used_for_compare,
                rtol=TestDualAscend.rel_tolerance_used_for_compare,
            )
        )

        logger.info(f"MAX_ERR_2: {abs(final_iterate.nu_array_2d - opt_nu_array_1d).max()}")
        assert final_iterate.nu_array_2d is not None
        logger.debug(final_iterate.nu_array_2d)
        logger.debug(opt_nu_array_1d)
        logger.debug(final_iterate.nu_array_2d - opt_nu_array_1d)
        logger.debug(allclose(final_iterate.nu_array_2d, opt_nu_array_1d))
        logger.debug(TestDualAscend.abs_tolerance_used_for_compare)
        logger.debug(TestDualAscend.rel_tolerance_used_for_compare)
        logger.debug(
            allclose(
                final_iterate.nu_array_2d,
                opt_nu_array_1d,
                atol=TestDualAscend.abs_tolerance_used_for_compare,
                rtol=TestDualAscend.rel_tolerance_used_for_compare,
            )
        )
        self.assertTrue(
            allclose(
                final_iterate.nu_array_2d,
                opt_nu_array_1d,
                atol=TestDualAscend.abs_tolerance_used_for_compare,
                rtol=TestDualAscend.rel_tolerance_used_for_compare,
            )
        )

    @classmethod
    def _get_simple_quad_problem(cls) -> OptProb:
        obj_fcn: QuadraticFunction = get_sum_of_square_function(cls.domain_dim)
        eq_cnst_fcn: AffineFunction = get_sum_function(cls.domain_dim, -1.0)
        return OptProb(obj_fcn, eq_cnst_fcn)

    @classmethod
    def _get_quad_problem_with_random_eq_cnsts(cls, num_eq_cnsts: int) -> OptProb:
        obj_fcn: QuadraticFunction = get_sum_of_square_function(cls.domain_dim)
        eq_cnst_fcn: AffineFunction = AffineFunction(
            randn(cls.domain_dim, num_eq_cnsts), randn(num_eq_cnsts)
        )
        return OptProb(obj_fcn, eq_cnst_fcn)


if __name__ == "__main__":
    unittest.main()
