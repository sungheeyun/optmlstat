"""
test for gradient descent method
"""

import logging  # noqa: F401
import os
from logging import getLogger, Logger

import click
import numpy as np
import numpy.random as nr
from freq_used.logging_utils import set_logging_basic_config
from freq_used.plotting import get_figure
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import linalg as la

from optmlstat.functions.basic_functions.log_sum_exp import LogSumExp
from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.functions.function_base import FunctionBase
from optmlstat.functions.example_functions import get_cvxopt_book_for_grad_method
from optmlstat.linalg.utils import get_random_pos_def_array
from optmlstat.opt.constants import LineSearchMethod
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.grad_descent import GradDescent
from optmlstat.plotting.opt_res_plotter import OptimizationResultPlotter

logger: Logger = getLogger()


def solve_and_draw(
    opt_prob: OptProb,
    opt_params: OptParams,
    initial_x_2d: np.ndarray,
    /,
    *,
    contour: bool,
    contour_xlim: tuple[float, float],
    contour_ylim: tuple[float, float],
    true_opt_val: float | None = None,
    verbose: bool = True,
    x_trajectory: bool = True,
) -> None:
    grad_descent: GradDescent = GradDescent(LineSearchMethod.BackTrackingLineSearch)
    opt_res: OptResults = grad_descent.solve(opt_prob, opt_params, initial_x_2d)

    figure: Figure = get_figure(
        1,
        1,
        axis_width=5.0,
        axis_height=5.0,
        top_margin=0.5,
        bottom_margin=0.5,
        vertical_padding=0.5,
    )
    (ax,) = figure.get_axes()

    optimization_result_plotter: OptimizationResultPlotter = OptimizationResultPlotter(opt_res)
    optimization_result_plotter.plot_primal_and_dual_objs(
        ax, linestyle="-", marker="o", markersize=1.0, true_opt_val=true_opt_val
    )

    if x_trajectory:
        optimization_result_plotter.animate_primal_sol(
            contour=contour, contour_xlim=contour_xlim, contour_ylim=contour_ylim, interval=0.0
        )


@click.command()
@click.argument("problem", type=str)
def main(problem: str) -> None:
    set_logging_basic_config(
        __file__, level=eval(f"logging.{os.environ.get('TEST_LOG_LEVEL', 'INFO')}")
    )

    num_data_points: int = 10
    data_lim: tuple[float, float] = -3.0, 3.0
    obj_fcn: FunctionBase
    num_vars: int = 2
    opt_params: OptParams = OptParams(
        0.1,
        1000,
        back_tracking_line_search_alpha=0.2,
        back_tracking_line_search_beta=0.9,
        tolerance_on_grad=0.01,
    )

    # type defs
    P: np.ndarray
    q: np.ndarray

    true_opt_val: float | None = None
    if problem == "cvxopt_book":
        obj_fcn = get_cvxopt_book_for_grad_method()
    elif problem == "lse":
        num_vars = 100
        num_terms: int = 10
        obj_fcn = LogSumExp([0.1 * nr.randn(num_terms, num_vars)], 0.1 * nr.randn(1, num_terms))
    elif problem == "quad":
        num_vars = 100
        P = get_random_pos_def_array(num_vars)
        q = nr.randn(num_vars)
        obj_fcn = QuadraticFunction(P[:, :, None], q[:, None], np.zeros(1))
        opt_params = OptParams(
            0.1,
            300,
            back_tracking_line_search_alpha=0.2,
            back_tracking_line_search_beta=0.9,
            tolerance_on_grad=10.0,
        )
        true_opt_val = -np.dot(la.lstsq(P, q)[0], q) / 4.0
    elif problem == "well-conditioned_quad":
        num_vars = 100
        P = get_random_pos_def_array(num_vars)
        q = nr.randn(num_vars)
        obj_fcn = QuadraticFunction(P[:, :, None], q[:, None], np.zeros(1))
        opt_params = OptParams(
            0.1,
            30,
            back_tracking_line_search_alpha=0.2,
            back_tracking_line_search_beta=0.9,
            tolerance_on_grad=10.0,
        )
        true_opt_val = -np.dot(la.lstsq(P, q)[0], q) / 4.0
    else:
        assert False, problem

    opt_prob: OptProb = OptProb(obj_fcn)
    initial_x_2d: np.ndarray = (data_lim[1] - data_lim[0]) * nr.rand(
        num_data_points, num_vars
    ) + data_lim[0]

    solve_and_draw(
        opt_prob,
        opt_params,
        initial_x_2d,
        contour=(num_vars == 2),
        contour_xlim=data_lim,
        contour_ylim=data_lim,
        true_opt_val=true_opt_val,
    )
    plt.show()


if __name__ == "__main__":
    main()
