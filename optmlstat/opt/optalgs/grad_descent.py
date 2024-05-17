"""
gradient descent method
"""

from logging import Logger, getLogger
from typing import Any

import numpy as np

from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.derivative_based_optalg_base import DerivativeBasedOptAlgBase
from optmlstat.opt.optalg_decorators import unconstrained_opt_solver

logger: Logger = getLogger()


class GradDescent(DerivativeBasedOptAlgBase):

    @unconstrained_opt_solver
    def _solve(
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
        return self._derivative_based_iter_solve(
            opt_prob,
            opt_param,
            verbose,
            initial_x_array_2d=initial_x_array_2d,
            initial_lambda_array_2d=initial_lambda_array_2d,
            initial_nu_array_2d=initial_nu_array_2d,
        )

    def check_stopping_criteria(
        self,
        search_directions: np.ndarray,
        jac: np.ndarray,
        hess: np.ndarray | None,
        opt_param: OptParams,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        assert jac is not None
        assert jac.shape[1] == 1, jac.shape
        assert search_directions is not None
        assert search_directions.ndim == 2, search_directions.shape
        assert search_directions.shape[0] == jac.shape[0], (search_directions.shape, jac.shape)
        info: dict[str, Any] = dict(
            grad_norm_squared=(jac.squeeze(axis=1) ** 2).sum(axis=1),
            tolerance_on_grad=opt_param.tolerance_on_grad,
        )
        return info["grad_norm_squared"] < opt_param.tolerance_on_grad, info

    @property
    def need_hessian(self) -> bool:
        return False

    def get_search_dir(
        self, opt_prob: OptProb, jac: np.ndarray, hess: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert hess is None, hess.__class__
        return -jac.squeeze(axis=1), np.ndarray((jac.shape[0], 0)), np.ndarray((jac.shape[0], 0))
