"""
feasible Newton's method for linearly equality constrained minimization
"""

from logging import Logger, getLogger

import numpy as np
from scipy import linalg

from optmlstat.functions.basic_functions.affine_function import AffineFunction
from optmlstat.functions.function_base import FunctionBase
from optmlstat.opt.optalg_decorators import linear_eq_cnst_solver, obj_and_eq_only_solver
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.newtons_method_base import NewtonsMethodBase
from optmlstat.linalg.utils import block_array, skinny_empty_array_2d

logger: Logger = getLogger()


class FeasibleNewtonsMethodForLinearEqConstProb(NewtonsMethodBase):

    @obj_and_eq_only_solver
    @linear_eq_cnst_solver
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
        feasible: bool = self.check_primela_eq_feasibility(opt_prob, initial_x_2d)
        assert feasible, feasible
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
        jac_3d: np.ndarray,
        hess_4d: np.ndarray | None,
        lambda_2d: np.ndarray,
        nu_2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert hess_4d is not None, hess_4d.__class__
        assert opt_prob.eq_cnst_fcn is not None

        eq_cnst_fcn: FunctionBase = opt_prob.eq_cnst_fcn
        assert isinstance(eq_cnst_fcn, AffineFunction), eq_cnst_fcn.__class__

        _A_array_2d: np.ndarray = eq_cnst_fcn.slope_2d.T
        jac_2d: np.ndarray = jac_3d.squeeze(axis=1)
        hess_array_3d: np.ndarray = hess_4d.squeeze(axis=1)

        kkt_sol_array_2d: np.ndarray = np.vstack(
            [
                linalg.solve(
                    block_array([[hess_array_3d[idx], _A_array_2d.T], [_A_array_2d, 0]]),
                    np.concatenate((-jac_1d, np.zeros(eq_cnst_fcn.num_outputs))),
                    assume_a="sym",
                )
                for idx, jac_1d in enumerate(jac_2d)
            ]
        )

        search_direction_2d, _nu_2d = (
            kkt_sol_array_2d[:, : opt_prob.dim_domain],
            kkt_sol_array_2d[:, opt_prob.dim_domain :],  # noqa:E203
        )

        directional_deriv_1d: np.ndarray = (search_direction_2d * jac_2d).sum(axis=1)

        # assign nu directly
        nu_2d -= nu_2d
        nu_2d += _nu_2d

        return (
            search_direction_2d,
            skinny_empty_array_2d(jac_3d.shape[0]),
            np.zeros((jac_3d.shape[0], opt_prob.num_eq_cnst)),
            directional_deriv_1d,
        )

    @property
    def stopping_criterion_name(self) -> str:
        return "newton_dec"
