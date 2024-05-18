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
from matplotlib import pyplot as plt

from optmlstat.functions.basic_functions.log_sum_exp import LogSumExp
from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.functions.function_base import FunctionBase
from optmlstat.functions.some_typical_functions import get_cvxopt_book_for_grad_method
from optmlstat.linalg.utils import get_random_pos_def_array
from optmlstat.opt.constants import LineSearchMethod
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.derivative_based_optalg_base import DerivativeBasedOptAlgBase
from optmlstat.opt.optalgs.grad_descent import GradDescent
from optmlstat.opt.optalgs.unconstrained_newtons_method import UnconstrainedNewtonsMethod
from optmlstat.plotting.opt_res_plotter import OptimizationResultPlotter

logger: Logger = getLogger()


def solve_and_draw(
    problem_name: str,
    algorithm: str,
    opt_prob: OptProb,
    opt_params: OptParams,
    verbose: bool,
    initial_x_2d: np.ndarray,
    /,
    *,
    proportional_real_solving_time,
) -> None:
    lsm: LineSearchMethod = LineSearchMethod.BackTrackingLineSearch
    unc_algorithm: DerivativeBasedOptAlgBase = UnconstrainedNewtonsMethod(lsm)
    if algorithm == "grad":
        unc_algorithm = GradDescent(lsm)
    elif algorithm == "newton":
        pass
    else:
        assert False, algorithm

    opt_res: OptResults = unc_algorithm.solve(opt_prob, opt_params, verbose, initial_x_2d)
    opt_res.result_analysis()

    OptimizationResultPlotter.standard_plotting(
        opt_res,
        f"problem: {problem_name}, algorithm: {algorithm}"
        + f", optimization time: {opt_res.solve_time:.3g} [sec]"
        + f", # opt vars: {opt_res.opt_prob.dim_domain}",
        proportional_real_solving_time=proportional_real_solving_time,
    )


@click.command()
@click.argument("problem", type=str)
@click.option("-v", "--verbose", is_flag=True, default=False, help="verbose optimization processes")
@click.option(
    "-r",
    "--proportional-real-solving-time",
    is_flag=True,
    default=False,
    help="trajectory animation speed proportional to reaal solving time",
)
@click.option(
    "-g",
    "--gradient",
    is_flag=True,
    default=False,
    help="use gradient descent instead of Newton's method",
)
@click.option(
    "-s",
    "--random-seed",
    is_flag=False,
    type=int,
    help="seed for random number generation",
)
def main(
    problem: str,
    gradient: bool,
    verbose: bool,
    proportional_real_solving_time: bool,
    random_seed: int | None,
) -> None:
    set_logging_basic_config(
        __file__, level=eval(f"logging.{os.environ.get('TEST_LOG_LEVEL', 'INFO')}")
    )
    if random_seed is not None:
        nr.seed(random_seed)

    num_data_points: int = 20
    data_lim: tuple[float, float] = -3.0, 3.0
    obj_fcn: FunctionBase
    num_vars: int = 2
    opt_params: OptParams = OptParams(
        0.1,
        100,
        back_tracking_line_search_alpha=0.2,
        back_tracking_line_search_beta=0.9,
        tolerance_on_grad=1e-2,
        tolerance_on_newton_dec=1e-2,
    )

    # type defs
    P: np.ndarray
    q: np.ndarray
    r: float

    if problem == "cvxopt-book":
        obj_fcn = get_cvxopt_book_for_grad_method()
    elif problem == "lse":
        num_vars = 100
        num_terms: int = 300
        obj_fcn = LogSumExp([1e-1 * nr.randn(num_terms, num_vars)], 1e-1 * nr.randn(1, num_terms))
        data_lim = -3.0, 4.0
    elif problem == "random-quad":
        num_vars = 100
        P = get_random_pos_def_array(num_vars)
        q = nr.randn(num_vars)
        r = 10.0
        obj_fcn = QuadraticFunction(P[:, :, None], q[:, None], r * np.ones(1))
        opt_params = OptParams(
            0.1,
            100,
            back_tracking_line_search_alpha=0.2,
            back_tracking_line_search_beta=0.9,
            tolerance_on_grad=1e-2,
            tolerance_on_newton_dec=1e-2,
        )
    elif problem == "well-conditioned-quad":
        num_vars = 100
        P = get_random_pos_def_array(np.logspace(-1.0, 1.0, num_vars))
        q = nr.randn(num_vars)
        r = 100.0
        obj_fcn = QuadraticFunction(P[:, :, None], q[:, None], r * np.ones(1))
        opt_params = OptParams(
            0.1,
            100,
            back_tracking_line_search_alpha=0.25,
            back_tracking_line_search_beta=0.5,
            tolerance_on_grad=1e1,
            tolerance_on_newton_dec=1e-6,
        )
    elif problem == "ill-conditioned-quad":
        num_vars = 100
        P = get_random_pos_def_array(np.logspace(-3.0, 3.0, num_vars))
        q = nr.randn(num_vars)
        r = 100.0
        obj_fcn = QuadraticFunction(P[:, :, None], q[:, None], r * np.ones(1))
        opt_params = OptParams(
            0.1,
            15,
            back_tracking_line_search_alpha=0.2,
            back_tracking_line_search_beta=0.9,
            tolerance_on_grad=1e-1,
            tolerance_on_newton_dec=1e-6,
        )
    else:
        assert False, problem

    opt_prob: OptProb = OptProb(obj_fcn, None, None)
    initial_x_2d: np.ndarray = (data_lim[1] - data_lim[0]) * nr.rand(
        num_data_points, num_vars
    ) + data_lim[0]

    solve_and_draw(
        problem,
        "grad" if gradient else "newton",
        opt_prob,
        opt_params,
        verbose,
        initial_x_2d,
        proportional_real_solving_time=proportional_real_solving_time,
    )
    plt.show()


if __name__ == "__main__":
    main()
