from typing import Optional
from logging import Logger, getLogger

from numpy import ndarray
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
        self.learning_rate: float = learning_rate

    def solve(
        self,
        opt_prob: OptimizationProblem,
        initial_x_2d: Optional[ndarray] = None,
        initial_y_2d: Optional[ndarray] = None,
    ) -> OptimizationResult:

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
        assert isinstance(obj_fcn, QuadraticFunction)
        assert obj_fcn.is_strictly_convex
        assert obj_fcn.num_outputs == 1

        assert isinstance(opt_prob.eq_cnst_fcn, AffineFunction)
        eq_cnst_fcn: AffineFunction = opt_prob.eq_cnst_fcn

        if initial_x_2d is None:
            initial_x_2d = randn(1, opt_prob.domain_dim)

        if initial_y_2d is None:
            initial_y_2d = randn(1, eq_cnst_fcn.num_outputs)

        assert initial_x_2d.shape[0] == initial_y_2d.shape[0]
        assert initial_x_2d.shape[1] == opt_prob.domain_dim
        assert initial_y_2d.shape[1] == eq_cnst_fcn.num_outputs

        x_1d: ndarray = initial_x_2d[0, :]
        y_1d: ndarray = initial_y_2d[0, :]

        for idx in range(100):
            logger.debug(x_1d.shape)
            logger.debug(y_1d.shape)
            logger.debug(obj_fcn.slope_array_2d[:, 0])
            logger.debug(eq_cnst_fcn.slope_array_2d.T)
            logger.debug(y_1d.dot(eq_cnst_fcn.slope_array_2d.T))
            x_1d = -0.5 * la.solve(
                obj_fcn.quad_array_3d[:, :, 0], obj_fcn.slope_array_2d[:, 0] + y_1d.dot(eq_cnst_fcn.slope_array_2d.T)
            )

            logger.info(f"x_1d: {x_1d}")

            logger.debug(x_1d)
            logger.debug(eq_cnst_fcn.get_y_values_2d(x_1d))

            y_1d += self.learning_rate * eq_cnst_fcn.get_y_values_2d(x_1d)

            logger.info(f"y_1d: {y_1d}")

        return OptimizationResult()
