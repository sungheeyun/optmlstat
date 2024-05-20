"""
Newton's method for unconstrained optimization
"""

from logging import Logger, getLogger

import numpy as np
from scipy import linalg

from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalg_decorators import (
    unconstrained_opt_solver,
)
from optmlstat.opt.optalgs.newtons_method_base import NewtonsMethodBase

logger: Logger = getLogger()


class UnconstrainedNewtonsMethod(NewtonsMethodBase):

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
        return super()._solve(
            opt_prob,
            opt_param,
            verbose,
            initial_x_2d,
            initial_lambda_2d=initial_lambda_2d,
            initial_nu_2d=initial_nu_2d,
        )

    def search_direction_and_update_lag_vars(
        self,
        opt_prob: OptProb,
        x_2d: np.ndarray,
        jac: np.ndarray,
        hess_4d: np.ndarray | None,
        lambda_2d: np.ndarray,
        nu_2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert hess_4d is not None, hess_4d.__class__
        jac_array_2d: np.ndarray = jac.squeeze(axis=1)
        hess_array_3d: np.ndarray = hess_4d.squeeze(axis=1)

        search_direction_2d: np.ndarray = np.vstack(
            [
                linalg.solve(hess_array_3d[idx], -jac_1d, assume_a="sym")
                for idx, jac_1d in enumerate(jac_array_2d)
            ]
        )

        assert opt_prob.obj_fcn is not None
        return (
            search_direction_2d,
            np.ndarray((jac.shape[0], 0)),
            np.ndarray((jac.shape[0], 0)),
            (search_direction_2d * jac_array_2d).sum(axis=1),
        )

    @property
    def stopping_criterion_name(self) -> str:
        return "newton_dec"
