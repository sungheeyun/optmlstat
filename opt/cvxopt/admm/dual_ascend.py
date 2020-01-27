from typing import Optional
from logging import Logger, getLogger

from numpy import ndarray, newaxis, allclose
from numpy.random import randn
import numpy.linalg as la

from functions.function_base import FunctionBase
from functions.basic_functions.affine_function import AffineFunction
from functions.basic_functions.quadratic_function import QuadraticFunction
from opt.opt_alg.optimization_algorithm_base import OptimizationAlgorithmBase
from opt.opt_prob.optimization_problem import OptimizationProblem
from opt.optimization_result import OptimizationResult


logger: Logger = getLogger()


class DualAscend(OptimizationAlgorithmBase):
    """
    Dual Ascend algorithm
    """

    def __init__(self, learning_rate: float):
        super(DualAscend, self).__init__(learning_rate)

    @property
    def solve(
        self,
        opt_prob: OptimizationProblem,
        initial_x_2d: Optional[ndarray] = None,
        initial_y_2d: Optional[ndarray] = None,
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

        initial_x_2d:
         N-by-n array representing initial points for x.

        initial_y_2d:
         N-by-1 array representing initial points for y.

        Returns
        -------
        optimization_result:
         OptimizationResult instance.
        """

        # TODO need to accept more parameters, such as, stopping criteria, such as, size of (Ax - b),
        #  maximum number of iteration, difference in x and y, etc.
        # TODO store optimization information to optimization result
        # TODO implement argmin_x part in FunctionBase, not in here.
        # TODO make this work for multi-input and multi-output

        assert opt_prob.is_convex
        assert opt_prob.ineq_cnst_fcn is None
        assert opt_prob.eq_cnst_fcn is not None and opt_prob.eq_cnst_fcn.is_affine

        obj_fcn: FunctionBase = opt_prob.obj_fcn
        assert obj_fcn is not None
        assert obj_fcn.is_convex
        assert obj_fcn.num_outputs == 1

        conjugate: QuadraticFunction = obj_fcn.conjugate
        assert conjugate.is_convex

        assert isinstance(opt_prob.eq_cnst_fcn, AffineFunction)
        eq_cnst_fcn: AffineFunction = opt_prob.eq_cnst_fcn

        if initial_x_2d is None and initial_y_2d is None:
            initial_x_2d = randn(1, opt_prob.domain_dim)
            initial_y_2d = randn(1, eq_cnst_fcn.num_outputs)
        else:
            assert initial_x_2d is not None and initial_y_2d is not None

        assert initial_x_2d.shape[0] == initial_y_2d.shape[0]
        assert initial_x_2d.shape[1] == opt_prob.domain_dim
        assert initial_y_2d.shape[1] == eq_cnst_fcn.num_outputs

        x_array_1d: ndarray = initial_x_2d[0, :]
        y_array_1d: ndarray = initial_y_2d[0, :]


        for idx in range(1000):
            logger.debug(x_array_1d.shape)
            logger.debug(y_array_1d.shape)
            logger.debug(obj_fcn.slope_array_2d[:, 0])
            logger.debug(eq_cnst_fcn.slope_array_2d.T)
            logger.debug(y_array_1d.dot(eq_cnst_fcn.slope_array_2d.T))

            nu_array_1d_for_x_update: ndarray = - y_array_1d.dot(eq_cnst_fcn.slope_array_2d.T)

            # x-minimization step
            x_array_1d: ndarray = obj_fcn.conjugate_arg(nu_array_1d_for_x_update[newaxis, :])[0, :, 0]

            logger.debug(f"x_1d: {x_array_1d}")
            logger.debug(x_array_1d)
            logger.debug(eq_cnst_fcn.get_y_values_2d(x_array_1d))

            # dual variable update
            y_array_1d += self.learning_rate * eq_cnst_fcn.get_y_values_2d(x_array_1d)

            nu_array_1d_for_dual_function_eval: ndarray = - y_array_1d.dot(eq_cnst_fcn.slope_array_2d.T)

            logger.debug(f"y_1d: {y_array_1d}")

            primal_fcn_val: float = obj_fcn.get_y_values_2d(x_array_1d[newaxis, :])[0, 0]
            dual_fcn_val: float = -conjugate.get_y_values_2d(nu_array_1d_for_dual_function_eval[newaxis, :])[0, 0] + y_array_1d.dot(eq_cnst_fcn.intercept_array_1d)

            logger.info(f"primal: {primal_fcn_val}")
            logger.info(f"dual: {dual_fcn_val}")
            logger.info(f"GAP: {primal_fcn_val - dual_fcn_val}")

        return OptimizationResult()
