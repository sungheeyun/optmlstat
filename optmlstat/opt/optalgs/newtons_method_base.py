"""
Newton's method for unconstrained optimization
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
    """
    gradient descent method
    """

    @twice_differentiable_obj_required_solver
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
        return self._iter_solve(
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
        assert search_directions is not None
        assert jac is not None
        assert hess is not None

        assert jac.ndim == 3, jac.shape
        assert jac.shape[1] == 1, jac.shape
        assert hess.ndim == 4, hess.shape
        assert hess.shape[1] == 1, hess.shape
        assert search_directions.ndim == 2, search_directions.shape
        assert search_directions.shape[0] == jac.shape[0], (search_directions.shape, jac.shape)

        hess_3d: np.ndarray = hess.squeeze(axis=1)

        info: dict[str, Any] = dict(
            newton_dec=np.array(
                [
                    np.dot(np.dot(hess_2d, search_directions[idx]), search_directions[idx]) / 2.0
                    for idx, hess_2d in enumerate(hess_3d)
                ]
            ),
            tolerance_on_newton_dec=opt_param.tolerance_on_newton_dec,
        )
        if opt_param.tolerance_on_newton_dec is None:
            return np.array([False] * jac.shape[0]), info

        return info["newton_dec"] < opt_param.tolerance_on_newton_dec, info

    @property
    def need_hessian(self) -> bool:
        return True

    @abstractmethod
    def get_search_dir(
        self, opt_prob: OptProb, jac: np.ndarray, hess: np.ndarray | None
    ) -> np.ndarray:
        pass
