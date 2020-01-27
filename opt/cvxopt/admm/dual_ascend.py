from typing import Optional
from logging import Logger, getLogger

from numpy import ndarray, newaxis

from functions.function_base import FunctionBase
from functions.basic_functions.affine_function import AffineFunction
from functions.basic_functions.quadratic_function import QuadraticFunction
from opt.opt_alg.optimization_algorithm_base import OptimizationAlgorithmBase
from opt.opt_prob import OptimizationProblem
from opt.opt_res import OptimizationResult
from opt.opt_prob_eval import OptimizationProblemEvaluation
from opt.opt_decorators import solver, single_obj_solver, eq_cnst_solver, linear_eq_cnst_solver, convex_solver
from opt.iteration import Iteration


logger: Logger = getLogger()


class DualAscend(OptimizationAlgorithmBase):
    """
    Dual Ascend algorithm
    """

    def __init__(self, learning_rate: float):
        super(DualAscend, self).__init__(learning_rate)

    @convex_solver
    @linear_eq_cnst_solver
    @single_obj_solver
    @eq_cnst_solver
    @solver
    def solve(
        self,
        opt_prob: OptimizationProblem,
        initial_x_array_2d: Optional[ndarray] = None,
        initial_lambda_array_2d: Optional[ndarray] = None,
        initial_nu_array_2d: Optional[ndarray] = None,
    ) -> OptimizationResult:
        """
        This only deals with a single objective optimization problem with one linear equality constraint
        with no inequality constraint where the objective function should be convex and bounded below.
        The objective function should properly implement its methods, conjugate and conjugate_arg
        since this function depends on the correct behavior of those functions.

        This only deals with the function which as only one output.
        To parallelize the optimization process for multiple outputs,
        the client should consider a parallelization procedure wrapping this method,
        which does not depend on the details of this method.

        This will, however, deal with multiple x trajectories which start from different initial points.
        Thus you can test how the optimization process proceeds from different initial points
        simultaneously.

        Parameters
        ----------
        opt_prob:
         The optimization problem to solve.
        initial_x_array_2d:
         N-by-n array representing initial points for x.
        initial_lambda_array_2d:
         N-by-m array representing initial points for Lagrange multipliers for inequality constraints.
        initial_nu_array_2d:
         N-by-p array representing initial points for Lagrange multipliers for equality constraints.

        Returns
        -------
        optimization_result:
         OptimizationResult instance.
        """

        # TODO (4) need to accept more parameters, such as, stopping criteria, such as, size of (Ax - b),
        #  maximum number of iteration, difference in x and y, etc.

        obj_fcn: FunctionBase = opt_prob.obj_fcn
        eq_cnst_fcn: AffineFunction = opt_prob.eq_cnst_fcn

        assert initial_x_array_2d is not None
        assert initial_nu_array_2d is not None

        conjugate: QuadraticFunction = obj_fcn.conjugate
        assert conjugate.is_convex

        opt_res: OptimizationResult = OptimizationResult(opt_prob, self)

        opt_res.register_solution(iteration=Iteration(0), primal_prob_evaluation=opt_prob.evaluate(initial_x_array_2d))

        y_array_2d: ndarray = initial_nu_array_2d.copy()

        for idx in range(1000):
            iteration = Iteration(idx + 1)

            nu_array_2d_for_x_update: ndarray = -y_array_2d.dot(eq_cnst_fcn.slope_array_2d.T)

            # x-minimization step
            x_array_2d: ndarray = obj_fcn.conjugate_arg(nu_array_2d_for_x_update)[:, :, 0]

            # dual variable update
            y_array_2d += self.learning_rate * eq_cnst_fcn.get_y_values_2d(x_array_2d)

            nu_array_2d_for_dual_function_eval: ndarray = -y_array_2d.dot(eq_cnst_fcn.slope_array_2d.T)
            dual_fcn_array_2d: ndarray = -conjugate.get_y_values_2d(
                nu_array_2d_for_dual_function_eval
            ) + y_array_2d.dot(eq_cnst_fcn.intercept_array_1d[:, newaxis])

            opt_res.register_solution(
                iteration=iteration,
                primal_prob_evaluation=opt_prob.evaluate(x_array_2d),
                dual_prob_evaluation=OptimizationProblemEvaluation(
                    opt_prob=opt_prob, x_array_2d=y_array_2d, obj_fcn_array_2d=dual_fcn_array_2d
                ),
            )

        return opt_res
