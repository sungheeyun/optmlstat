"""
base class for classes of Newton's methods for diverse types of problems such as unconstrained,
linearly eq constrained only, linearly eq constrained and general ineq constrained, etc.
"""

from abc import abstractmethod
from logging import Logger, getLogger
from typing import Any

import numpy as np

from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalg_decorators import (
    twice_differentiable_obj_required_solver,
)
from optmlstat.opt.optalgs.derivative_based_optalg_base import DerivativeBasedOptAlgBase

logger: Logger = getLogger()


class NewtonsMethodBase(DerivativeBasedOptAlgBase):

    @twice_differentiable_obj_required_solver
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
        self, opt_param: OptParams, directional_deriv: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        assert directional_deriv.ndim == 1, directional_deriv.shape
        info: dict[str, Any] = dict(
            newton_dec=-directional_deriv,
            tolerance_on_newton_dec=opt_param.tolerance_on_newton_dec,
        )
        return info["newton_dec"] < opt_param.tolerance_on_newton_dec, info

    @property
    def need_hessian(self) -> bool:
        return True

    @abstractmethod
    def search_direction_and_update_lag_vars(
        self,
        opt_prob: OptProb,
        jac: np.ndarray,
        hess_4d: np.ndarray | None,
        lambda_2d: np.ndarray,
        nu_2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass
