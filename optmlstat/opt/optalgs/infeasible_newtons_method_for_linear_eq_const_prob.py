"""
infeasible Newton's method for linearly equality constrained minimization
"""

from logging import Logger, getLogger
from typing import Callable

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


class InfeasibleNewtonsMethodForLinearEqConstProb(NewtonsMethodBase):

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

        _a_2d: np.ndarray = eq_cnst_fcn.a_2d
        _a_2d_T: np.ndarray = _a_2d.T
        _b_1d: np.ndarray = -eq_cnst_fcn.b_1d
        jac_2d: np.ndarray = jac_3d.squeeze(axis=1)
        hess_3d: np.ndarray = hess_4d.squeeze(axis=1)

        # residuals: (# members) - by - (# opt vars + # eq cnsts)
        residuals_2d: np.ndarray = np.array(
            [
                np.hstack(
                    (
                        jac_2d[member_idx] + np.dot(_a_2d_T, nu_2d[member_idx, :]),
                        np.dot(_a_2d, x_1d) - _b_1d,
                    )
                )
                for member_idx, x_1d in enumerate(x_2d)
            ]
        )

        search_direction_x_nu_2d: np.ndarray = np.array(
            [
                linalg.solve(
                    block_array([[hess_2d, _a_2d_T], [_a_2d, 0.0]]), -residuals_2d[member_idx]
                )
                for member_idx, hess_2d in enumerate(hess_3d)
            ]
        )

        search_direction_x_2d: np.ndarray = search_direction_x_nu_2d[:, : opt_prob.dim_domain]
        search_direction_lambda_2d: np.ndarray = skinny_empty_array_2d(x_2d.shape[0])
        search_direction_nu_2d: np.ndarray = search_direction_x_nu_2d[
            :, opt_prob.dim_domain :  # noqa:E203
        ]

        directional_derivative_1d: np.ndarray = -np.sqrt((residuals_2d**2).sum(axis=1))

        return (
            search_direction_x_2d,
            search_direction_lambda_2d,
            search_direction_nu_2d,
            directional_derivative_1d,
        )

    def line_search_loss_fcn(self, opt_prob: OptProb) -> Callable:

        assert isinstance(opt_prob.eq_cnst_fcn, AffineFunction), opt_prob.eq_cnst_fcn.__class__

        _a_2d: np.ndarray = opt_prob.eq_cnst_fcn.a_2d
        _a_2d_T: np.ndarray = _a_2d.T
        _b_1d: np.ndarray = -opt_prob.eq_cnst_fcn.b_1d

        def loss_fcn(x_lambda_nu_2d: np.ndarray) -> np.ndarray:
            x_2d: np.ndarray = x_lambda_nu_2d[:, : opt_prob.dim_domain]
            nu_2d: np.ndarray = x_lambda_nu_2d[:, opt_prob.dim_domain :]  # noqa:E203
            assert opt_prob.obj_fcn is not None
            jac_2d: np.ndarray = opt_prob.obj_fcn.jacobian(x_2d).squeeze(axis=1)

            residuals_2d: np.ndarray = np.array(
                [
                    np.hstack(
                        (
                            jac_2d[member_idx] + np.dot(_a_2d_T, nu_2d[member_idx, :]),
                            np.dot(_a_2d, x_1d) - _b_1d,
                        )
                    )
                    for member_idx, x_1d in enumerate(x_2d)
                ]
            )
            return np.sqrt((residuals_2d**2).sum(axis=1, keepdims=True))

        return loss_fcn

    @property
    def stopping_criterion_name(self) -> str:
        return "residual_norm"
