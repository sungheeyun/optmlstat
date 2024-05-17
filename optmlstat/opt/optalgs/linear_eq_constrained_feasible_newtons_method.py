"""
Newton's method for unconstrained optimization
"""

from logging import Logger, getLogger

import numpy as np
from scipy import linalg

from optmlstat.functions.basic_functions.affine_function import AffineFunction
from optmlstat.functions.function_base import FunctionBase
from optmlstat.opt.optalg_decorators import linear_eq_cnst_solver, eq_cnst_solver
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.newtons_method_base import NewtonsMethodBase
from optmlstat.linalg.utils import block_array

logger: Logger = getLogger()


class LinearEqConstrainedFeasibleNewtonsMethod(NewtonsMethodBase):
    """
    gradient descent method
    """

    @eq_cnst_solver
    @linear_eq_cnst_solver
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
        feasible: bool = self.check_primela_eq_feasibility(opt_prob, initial_x_array_2d)
        assert feasible, feasible
        return super()._solve(
            opt_prob,
            opt_param,
            verbose,
            initial_x_array_2d=initial_x_array_2d,
            initial_lambda_array_2d=initial_lambda_array_2d,
            initial_nu_array_2d=initial_nu_array_2d,
        )

    def get_search_dir(
        self, opt_prob: OptProb, jac: np.ndarray, hess: np.ndarray | None
    ) -> np.ndarray:
        assert hess is not None, hess.__class__
        assert opt_prob.eq_cnst_fcn is not None

        eq_cnst_fcn: FunctionBase = opt_prob.eq_cnst_fcn
        assert isinstance(eq_cnst_fcn, AffineFunction), eq_cnst_fcn.__class__

        _A_array_2d: np.ndarray = eq_cnst_fcn.slope_array_2d.T
        jac_array_2d: np.ndarray = jac.squeeze(axis=1)
        hess_array_3d: np.ndarray = hess.squeeze(axis=1)

        return np.vstack(
            [
                linalg.solve(
                    block_array([[hess_array_3d[idx], _A_array_2d.T], [_A_array_2d, 0]]),
                    np.concatenate((-jac_1d, np.zeros(eq_cnst_fcn.num_outputs))),
                    assume_a="sym",
                )[: opt_prob.dim_domain]
                for idx, jac_1d in enumerate(jac_array_2d)
            ]
        )
