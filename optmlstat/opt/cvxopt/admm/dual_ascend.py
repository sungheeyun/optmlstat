"""
dual ascend as described in ADMM paper by Boyd
"""

from logging import Logger, getLogger

import numpy as np
from numpy import ndarray, newaxis

from optmlstat.functions.function_base import FunctionBase
from optmlstat.functions.basic_functions.affine_function import AffineFunction
from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.opt.optalgs.optalg_base import OptAlgBase
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.opt_prob_eval import OptProbEval
from optmlstat.opt.optalg_decorators import (
    solver,
    single_obj_solver,
    obj_and_eq_only_solver,
    linear_eq_cnst_solver,
    convex_solver,
)
from optmlstat.opt.iteration import Iteration
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.learning_rate.learning_rate_strategy import LearningRateStrategy


logger: Logger = getLogger()


class DualAscend(OptAlgBase):

    @convex_solver
    @linear_eq_cnst_solver
    @single_obj_solver
    @obj_and_eq_only_solver
    @solver
    def _solve(
        self,
        opt_prob: OptProb,
        opt_param: OptParams,
        verbose: bool,
        /,
        *,
        initial_x_2d: np.ndarray,
        initial_lambda_2d: np.ndarray | None = None,
        initial_nu_2d: np.ndarray | None = None,
    ) -> OptResults:
        """
        This only deals with a single objective optimization problem
        with one linear equality constraint
        with no inequality constraint where the objective function
        should be convex and bounded below.
        The objective function should properly implement its methods, conjugate and conjugate_arg
        since this function depends on the correct behavior of those functions.

        This only deals with the function which has only one output.
        To parallelize the optimization process for multiple outputs,
        the client should consider a parallelization procedure wrapping this method,
        which does not depend on the details of this method.

        This will, however, deal with multiple x trajectories which start
        from different initial points.
        Thus you can test how the optimization process proceeds from different initial points
        simultaneously.

        Parameters
        ----------
        opt_prob:
         The optimization problem to solve.
        opt_param:
         Optimization parameter
        verbose:
         verbose
        initial_x_2d:
         N-by-n array representing initial points for x.
        initial_lambda_2d:
         N-by-m array representing initial points for Lagrange multipliers
         for inequality constraints.
        initial_nu_2d:
         N-by-p array representing initial points for Lagrange multipliers
         for equality constraints.

        Returns
        -------
        optimization_result:
         OptimizationResult instance.
        """

        dual_problem: OptProb = opt_prob.dual_problem
        obj_fcn: FunctionBase | None = opt_prob.obj_fcn
        eq_cnst_fcn: FunctionBase | None = opt_prob.eq_cnst_fcn

        assert obj_fcn is not None
        assert isinstance(eq_cnst_fcn, AffineFunction), eq_cnst_fcn.__class__

        assert initial_x_2d is not None
        assert initial_nu_2d is not None
        assert initial_lambda_2d is None
        initial_lambda_2d = np.ndarray((initial_x_2d.shape[0], 0))

        conjugate: FunctionBase = obj_fcn.conjugate
        assert isinstance(conjugate, QuadraticFunction), conjugate.__class__
        assert conjugate.is_convex

        opt_res: OptResults = OptResults(opt_prob, self)

        opt_res.register_solution(
            Iteration(0),
            opt_prob.evaluate(initial_x_2d),
            opt_prob.dual_problem.evaluate(np.hstack((initial_lambda_2d, initial_nu_2d))),
            verbose,
        )

        y_array_2d: ndarray = initial_nu_2d.copy()

        learning_rate_strategy: LearningRateStrategy = opt_param.learning_rate_strategy
        for idx in range(opt_param.max_num_outer_iterations):
            iteration = Iteration(idx + 1)

            nu_array_2d_for_x_update: ndarray = -y_array_2d.dot(eq_cnst_fcn.slope_2d.T)

            # x-minimization step
            x_array_2d: ndarray = obj_fcn.conjugate_arg(nu_array_2d_for_x_update)[:, :, 0]

            # dual variable update
            y_array_2d += learning_rate_strategy.get_learning_rate(
                iteration
            ) * eq_cnst_fcn.get_y_values_2d(x_array_2d)

            nu_array_2d_for_dual_function_eval: ndarray = -y_array_2d.dot(eq_cnst_fcn.slope_2d.T)
            dual_fcn_array_2d: ndarray = -conjugate.get_y_values_2d(
                nu_array_2d_for_dual_function_eval
            ) + y_array_2d.dot(eq_cnst_fcn.intercept_1d[:, newaxis])

            opt_res.register_solution(
                iteration,
                opt_prob.evaluate(x_array_2d),
                OptProbEval(opt_prob=dual_problem, x_2d=y_array_2d, obj_fcn_2d=dual_fcn_array_2d),
                verbose,
            )

        return opt_res
