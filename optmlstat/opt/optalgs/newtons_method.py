"""
Newton's method for unconstrained optimization
"""

from logging import Logger, getLogger

import numpy as np
from scipy import linalg

from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.unconstrained_optalg_base import UnconstrainedOptAlgBase
from optmlstat.opt.opt_alg_decorators import twice_differentiable_obj_required_solver

logger: Logger = getLogger()


class NewtonsMethod(UnconstrainedOptAlgBase):
    """
    gradient descent method
    """

    @twice_differentiable_obj_required_solver
    def solve(
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
        return self._solve(
            opt_prob,
            opt_param,
            verbose,
            initial_x_array_2d=initial_x_array_2d,
            initial_lambda_array_2d=initial_lambda_array_2d,
            initial_nu_array_2d=initial_nu_array_2d,
        )

    def satisfy_stopping_criteria(
        self, jac: np.ndarray, hess: np.ndarray | None, opt_param: OptParams
    ) -> np.ndarray:
        assert jac is not None
        assert jac.shape[1] == 1, jac.shape
        return np.sqrt((jac.squeeze(axis=1) ** 2).sum(axis=1)) < opt_param.tolerance_on_grad

    @property
    def need_hessian(self) -> bool:
        return True

    def get_search_dir(self, jac: np.ndarray, hess: np.ndarray | None) -> np.ndarray:
        assert hess is not None, hess.__class__
        jac_array_2d: np.ndarray = jac.squeeze(axis=1)
        hess_array_3d: np.ndarray = hess.squeeze(axis=1)

        return np.vstack(
            [
                linalg.solve(hess_array_3d[idx], -jac_1d, assume_a="sym")
                for idx, jac_1d in enumerate(jac_array_2d)
            ]
        )
