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
from optmlstat.functions.example_functions import get_cvxopt_book_for_grad_method
from optmlstat.functions.function_base import FunctionBase
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
    verbose: bool,
    trajectory: bool,
    initial_x_2d: np.ndarray,
    /,
    *,
    contour: bool,
    contour_xlim: tuple[float, float],
    contour_ylim: tuple[float, float],
    true_opt_val: float | None = None,
) -> None:
    grad_descent: GradDescent = GradDescent(LineSearchMethod.BackTrackingLineSearch)
    opt_res: OptResults = grad_descent.solve(
        opt_prob, opt_params, verbose, initial_x_array_2d=initial_x_2d
    )
    opt_res.result_analysis(true_opt_val)

    figure: Figure = get_figure(
        1,
        2,
        axis_width=5.0,
        axis_height=5.0,
        top_margin=0.5,
        bottom_margin=0.5,
        vertical_padding=0.5,
    )
    ax, gap_ax = figure.get_axes()

    optimization_result_plotter: OptimizationResultPlotter = OptimizationResultPlotter(opt_res)
    optimization_result_plotter.plot_primal_and_dual_objs(
        ax,
        gap_ax,
        true_opt_val,
        linestyle="-",
        marker="o",
        markersize=100.0 / np.array(opt_res.num_iterations_list).mean(),
    )

    if trajectory:
        optimization_result_plotter.animate_primal_sol(
            contour=contour,
            contour_xlim=contour_xlim,
            contour_ylim=contour_ylim,
            interval=3e3 / np.array(opt_res.num_iterations_list).mean(),
        )


@click.command()
@click.argument("problem", type=str)
@click.option("-b", "--verbose", is_flag=True, default=False, help="verbose optimization processes")
@click.option(
    "-t",
    "--trajectory",
    is_flag=True,
    default=False,
    help="show animation of optimization variable trajectory",
)
def main(problem: str, verbose: bool, trajectory: bool) -> None:
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
        tolerance_on_grad=1e-3,
    )

    # type defs
    P: np.ndarray
    q: np.ndarray
    r: float

    true_opt_val: float | None = None
    if problem == "cvxopt_book":
        obj_fcn = get_cvxopt_book_for_grad_method()
    elif problem == "lse":
        num_vars = 10
        num_terms: int = 5
        obj_fcn = LogSumExp([1e-1 * nr.randn(num_terms, num_vars)], 1e-1 * nr.randn(1, num_terms))
    elif problem == "quad":
        num_vars = 100
        P = get_random_pos_def_array(num_vars)
        q = nr.randn(num_vars)
        r = 10.0
        obj_fcn = QuadraticFunction(P[:, :, None], q[:, None], r * np.ones(1))
        opt_params = OptParams(
            0.1,
            50,
            back_tracking_line_search_alpha=0.2,
            back_tracking_line_search_beta=0.9,
            tolerance_on_grad=10.0,
        )
        true_opt_val = -np.dot(la.lstsq(P, q)[0], q) / 4.0 + r
    elif problem == "conditioned-quad":
        num_vars = 100
        P = get_random_pos_def_array(np.logspace(-2, 2, num_vars))
        q = nr.randn(num_vars)
        r = 100.0
        obj_fcn = QuadraticFunction(P[:, :, None], q[:, None], r * np.ones(1))
        opt_params = OptParams(
            0.1,
            300,
            back_tracking_line_search_alpha=0.2,
            back_tracking_line_search_beta=0.9,
            tolerance_on_grad=1e0,
        )
        true_opt_val = -np.dot(la.lstsq(P, q)[0], q) / 4.0 + r
    else:
        assert False, problem

    opt_prob: OptProb = OptProb(obj_fcn)
    initial_x_2d: np.ndarray = (data_lim[1] - data_lim[0]) * nr.rand(
        num_data_points, num_vars
    ) + data_lim[0]

    solve_and_draw(
        opt_prob,
        opt_params,
        verbose,
        trajectory,
        initial_x_2d,
        contour=(num_vars == 2),
        contour_xlim=data_lim,
        contour_ylim=data_lim,
        true_opt_val=true_opt_val,
    )
    plt.show()


if __name__ == "__main__":
    main()
