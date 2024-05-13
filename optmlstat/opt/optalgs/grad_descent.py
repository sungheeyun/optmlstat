"""
gradient descent method
"""

from logging import Logger, getLogger

import numpy as np

from optmlstat.functions.function_base import FunctionBase
from optmlstat.opt.constants import LineSearchMethod
from optmlstat.opt.iteration import Iteration
from optmlstat.opt.opt_alg_decorators import (
    solver,
    single_obj_solver,
    unconstrained_opt_solver,
    differentiable_obj_required_solver,
)
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.back_tracking_ls import BackTrackingLineSearch
from optmlstat.opt.optalgs.line_search_base import LineSearchBase
from optmlstat.opt.optalgs.optalg_base import OptAlgBase

logger: Logger = getLogger()


class GradDescent(OptAlgBase):
    """
    Dual Ascend algorithm
    """

    def __init__(self, line_search_method: LineSearchMethod) -> None:
        self.line_search_method: LineSearchMethod = line_search_method

    @solver
    @single_obj_solver
    @unconstrained_opt_solver
    @differentiable_obj_required_solver
    def solve(
        self,
        opt_prob: OptProb,
        opt_param: OptParams,
        initial_x_array_2d: np.ndarray | None = None,
        initial_lambda_array_2d: np.ndarray | None = None,
        initial_nu_array_2d: np.ndarray | None = None,
    ) -> OptResults:
        """
        gradient descent method
        """

        assert opt_param.back_tracking_line_search_alpha is not None
        assert opt_param.back_tracking_line_search_beta is not None

        line_search: LineSearchBase
        if self.line_search_method == LineSearchMethod.BackTrackingLineSearch:
            alpha: float = opt_param.back_tracking_line_search_alpha
            beta: float = opt_param.back_tracking_line_search_beta

            assert 0.0 < alpha < 0.5, alpha
            assert 0.0 < beta < 1.0, beta

            line_search = BackTrackingLineSearch(alpha, beta)
        else:
            assert False, self.line_search_method

        assert initial_x_array_2d is not None
        obj_fcn: FunctionBase | None = opt_prob.obj_fcn
        assert obj_fcn is not None
        opt_res: OptResults = OptResults(opt_prob, self)

        jac: np.ndarray = obj_fcn.jacobian(initial_x_array_2d)
        terminated: np.ndarray = self.satisfy_stopping_criteria(jac, opt_param)

        opt_res.register_solution(
            iteration=Iteration(0),
            primal_prob_evaluation=opt_prob.evaluate(initial_x_array_2d),
            terminated=terminated,
        )

        x_array_2d: np.ndarray = initial_x_array_2d
        for idx in range(opt_param.max_num_outer_iterations):
            iteration = Iteration(idx + 1)
            search_dir = -jac.squeeze(axis=1)

            t_array_1d: np.ndarray = line_search.search(obj_fcn, x_array_2d, search_dir)

            x_array_2d += t_array_1d[:, None] * search_dir

            jac = obj_fcn.jacobian(x_array_2d)
            terminated = self.satisfy_stopping_criteria(jac, opt_param)
            opt_res.register_solution(
                iteration=iteration,
                primal_prob_evaluation=opt_prob.evaluate(x_array_2d),
                terminated=terminated,
            )

            if (~terminated).sum() == 0:
                break

        return opt_res

    @staticmethod
    def satisfy_stopping_criteria(jac: np.ndarray, opt_param: OptParams) -> np.ndarray:
        assert jac is not None
        assert jac.shape[1] == 1, jac.shape
        # print(np.sqrt((jac.squeeze(axis=1) ** 2).sum(axis=1)))
        return np.sqrt((jac.squeeze(axis=1) ** 2).sum(axis=1)) < opt_param.tolerance_on_grad
