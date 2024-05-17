"""
Newton's method for unconstrained optimization
"""

from logging import Logger, getLogger

import numpy as np
from scipy import linalg

from optmlstat.functions.function_base import FunctionBase
from optmlstat.opt.optalg_decorators import (
    unconstrained_opt_solver,
)
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.newtons_method_base import NewtonsMethodBase

logger: Logger = getLogger()


class UnconstrainedNewtonsMethod(NewtonsMethodBase):

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
        return super()._solve(
            opt_prob,
            opt_param,
            verbose,
            initial_x_array_2d=initial_x_array_2d,
            initial_lambda_array_2d=initial_lambda_array_2d,
            initial_nu_array_2d=initial_nu_array_2d,
        )

    def loss_fcn_and_directional_deriv(
        self, opt_prob: OptProb, jac: np.ndarray, hess: np.ndarray | None
    ) -> tuple[FunctionBase, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert hess is not None, hess.__class__
        jac_array_2d: np.ndarray = jac.squeeze(axis=1)
        hess_array_3d: np.ndarray = hess.squeeze(axis=1)

        return (
            np.vstack(
                [
                    linalg.solve(hess_array_3d[idx], -jac_1d, assume_a="sym")
                    for idx, jac_1d in enumerate(jac_array_2d)
                ]
            ),
            np.ndarray((jac.shape[0], 0)),
            np.ndarray((jac.shape[0], 0)),
        )
