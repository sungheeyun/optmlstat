"""
base class for derivative based opt algorithms
"""

from abc import abstractmethod
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
from optmlstat.opt.optalgs.iterative_optalg_base import IterativeOptAlgBase
from optmlstat.opt.optalgs.line_search_base import LineSearchBase


class DerivativeBasedOptAlgBase(IterativeOptAlgBase):

    def __init__(self, line_search_method: LineSearchMethod) -> None:
        self.line_search_method: LineSearchMethod = line_search_method

    @solver
    @single_obj_solver
    @differentiable_obj_required_solver
    def _derivative_based_iter_solve(
        self,
        opt_prob: OptProb,
        opt_param: OptParams,
        verbose: bool,
        initial_x_2d: np.ndarray,
        /,
        *,
        initial_lambda_2d: np.ndarray | None = None,
        initial_nu_2d: np.ndarray | None = None,
    ) -> OptResults:
        assert initial_x_2d is not None

        x_2d: np.ndarray = initial_x_2d.copy()
        lambda_2d: np.ndarray = (
            np.zeros((initial_x_2d.shape[0], opt_prob.num_ineq_cnst))
            if initial_lambda_2d is None
            else initial_lambda_2d.copy()
        )
        nu_2d: np.ndarray = (
            np.zeros((initial_x_2d.shape[0], opt_prob.num_eq_cnst))
            if initial_nu_2d is None
            else initial_nu_2d.copy()
        )

        line_search: LineSearchBase
        if self.line_search_method == LineSearchMethod.BackTrackingLineSearch:
            line_search = BackTrackingLineSearch(
                opt_param.back_tracking_line_search_alpha, opt_param.back_tracking_line_search_beta
            )
        else:
            assert False, self.line_search_method

        assert opt_prob.obj_fcn is not None

        obj_fcn: FunctionBase = opt_prob.obj_fcn
        opt_res: OptResults = OptResults(opt_prob, self)
        dual_problem: OptProb = opt_prob.dual_problem

        loss_fcn: FunctionBase = self.line_search_loss_fcn(opt_prob)
        for idx in range(opt_param.max_num_outer_iterations + 1):
            jac_3d: np.ndarray = obj_fcn.jacobian(x_2d)
            hess_4d: np.ndarray | None = obj_fcn.hessian(x_2d) if self.need_hessian else None

            (
                search_direction_x_2d,
                search_direction_lambda_2d,
                search_direction_nu_2d,
                directional_deriv_1d,
            ) = self.search_direction_and_update_lag_vars(
                opt_prob, jac_3d, hess_4d, lambda_2d, nu_2d
            )

            terminated, stopping_criteria_info = self.check_stopping_criteria(
                opt_param, directional_deriv_1d
            )

            opt_res.register_solution(
                Iteration(idx),
                opt_prob.evaluate(x_2d),
                dual_problem.evaluate(np.hstack((lambda_2d, nu_2d))),
                verbose,
                terminated=terminated,
                stopping_criteria_info=stopping_criteria_info,
            )

            if (~terminated).sum() == 0:
                break

            # TODO (H) written on 16-May-2024
            #  change code so that it does not search for members who have already satisfied
            #  stopping criteria
            t_array_1d: np.ndarray = line_search.search(
                loss_fcn,
                np.hstack((x_2d, lambda_2d, nu_2d)),
                np.hstack(
                    (search_direction_x_2d, search_direction_lambda_2d, search_direction_nu_2d)
                ),
                directional_deriv_1d,
            )

            # updatea primal & dual variables

            t_array_2d: np.ndarray = t_array_1d[:, np.newaxis]
            x_2d += t_array_2d * search_direction_x_2d
            lambda_2d += t_array_2d * search_direction_lambda_2d
            nu_2d += t_array_2d * search_direction_nu_2d

        return opt_res

    @property
    @abstractmethod
    def need_hessian(self) -> bool:
        pass

    @abstractmethod
    def search_direction_and_update_lag_vars(
        self,
        opt_prob: OptProb,
        jac_3d: np.ndarray,
        hess_4d: np.ndarray | None,
        lambda_2d: np.ndarray,
        nu_2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        calculate search directions (and other related quantities)

        :param opt_prob:
        :param jac_3d: jacobians of obj fcn
        :param hess_4d: hessians of obj fcn
        :param nu_2d:
        :param lambda_2d:
        :return:
            loss_fcn
            search direction x - 2d array
            search direction lambda - 2d array
            search direction nu - 2d array
            directional derivative - 1d array
        """
        pass

    @abstractmethod
    def check_stopping_criteria(
        self,
        opt_param: OptParams,
        directional_derivative: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        pass

    def line_search_loss_fcn(self, opt_prob: OptProb) -> FunctionBase:
        assert opt_prob.obj_fcn is not None
        return opt_prob.obj_fcn
