"""
gradient descent method
"""

from abc import abstractmethod
from logging import Logger, getLogger
from typing import Any

import numpy as np

from optmlstat.functions.function_base import FunctionBase
from optmlstat.opt.constants import LineSearchMethod
from optmlstat.opt.iteration import Iteration
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalg_decorators import (
    solver,
    single_obj_solver,
    differentiable_obj_required_solver,
)
from optmlstat.opt.optalgs.back_tracking_ls import BackTrackingLineSearch
from optmlstat.opt.optalgs.line_search_base import LineSearchBase
from optmlstat.opt.optalgs.optalg_base import OptAlgBase

logger: Logger = getLogger()


class DerivativeBasedOptAlgBase(OptAlgBase):
    """
    Dual Ascend algorithm
    """

    def __init__(self, line_search_method: LineSearchMethod) -> None:
        self.line_search_method: LineSearchMethod = line_search_method

    @solver
    @single_obj_solver
    @differentiable_obj_required_solver
    def _iter_solve(
        self,
        opt_prob: OptProb,
        opt_param: OptParams,
        verbose: bool,
        /,
        *,
        initial_x_array_2d: np.ndarray,
        initial_lambda_array_2d: np.ndarray | None = None,
        initial_nu_array_2d: np.ndarray | None = None,
    ) -> OptResults:
        """
        unconstrained optimization algorithm
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

        x_array_2d: np.ndarray = initial_x_array_2d
        for idx in range(opt_param.max_num_outer_iterations + 1):
            jac: np.ndarray = obj_fcn.jacobian(initial_x_array_2d)
            hess: np.ndarray | None = None
            if self.need_hessian:
                hess = obj_fcn.hessian(initial_x_array_2d)

            search_dir = self.get_search_dir(opt_prob, jac, hess)
            terminated, stopping_criteria_info = self.check_stopping_criteria(
                search_dir, jac, hess, opt_param
            )

            opt_res.register_solution(
                Iteration(idx),
                opt_prob.evaluate(x_array_2d),
                verbose,
                terminated=terminated,
                stopping_criteria_info=stopping_criteria_info,
            )

            if (~terminated).sum() == 0:
                break

            # TODO (H) written on 16-May-2024
            #  change code so that it does not search for members who have already satisfied
            #  stopping criteria
            t_array_1d: np.ndarray = line_search.search(obj_fcn, x_array_2d, search_dir)
            x_array_2d += t_array_1d[:, None] * search_dir

        return opt_res

    @property
    @abstractmethod
    def need_hessian(self) -> bool:
        pass

    @abstractmethod
    def get_search_dir(
        self, opt_prob: OptProb, jac: np.ndarray, hess: np.ndarray | None
    ) -> np.ndarray:
        pass

    @abstractmethod
    def check_stopping_criteria(
        self,
        search_directions: np.ndarray,
        jac: np.ndarray,
        hess: np.ndarray | None,
        opt_param: OptParams,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        pass