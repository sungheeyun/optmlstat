"""
gradient descent method
"""

from logging import Logger, getLogger
from typing import Any

import numpy as np

from optmlstat.linalg.utils import skinny_empty_array_2d
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalg_decorators import unconstrained_opt_solver
from optmlstat.opt.optalgs.derivative_based_optalg_base import DerivativeBasedOptAlgBase

logger: Logger = getLogger()


class GradDescent(DerivativeBasedOptAlgBase):

    @unconstrained_opt_solver
    def _solve(
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
        return self._derivative_based_iter_solve(
            opt_prob,
            opt_param,
            verbose,
            initial_x_2d,
            initial_lambda_2d=initial_lambda_2d,
            initial_nu_2d=initial_nu_2d,
        )

    def check_stopping_criteria(
        self,
        opt_param: OptParams,
        directional_deriv: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        assert directional_deriv.ndim == 1, directional_deriv.shape
        info: dict[str, Any] = dict(
            grad_norm_squared=-directional_deriv,
            tolerance_on_grad=opt_param.tolerance_on_grad,
        )
        return info["grad_norm_squared"] < opt_param.tolerance_on_grad, info

    @property
    def need_hessian(self) -> bool:
        return False

    def search_direction_and_update_lag_vars(
        self,
        opt_prob: OptProb,
        jac: np.ndarray,
        hess_4d: np.ndarray | None,
        lambda_2d: np.ndarray,
        nu_2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert hess_4d is None, hess_4d.__class__
        assert opt_prob.obj_fcn is not None

        num_members: int = jac.shape[0]
        assert hess_4d is None or hess_4d.shape[0] == num_members, (
            jac.shape,
            None if hess_4d is None else hess_4d.shape,  # type:ignore
        )

        return (
            -jac.squeeze(axis=1),
            skinny_empty_array_2d(num_members),
            skinny_empty_array_2d(num_members),
            -(jac.squeeze(axis=1) ** 2).sum(axis=1),
        )
