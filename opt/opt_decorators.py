from typing import Callable, Optional, Any
from logging import Logger, getLogger
from functools import wraps

from numpy import ndarray
from numpy.random import randn

from opt.opt_prob.optimization_problem import OptimizationProblem
from opt.optimization_result import OptimizationResult

logger: Logger = getLogger()


def solver(func: Callable) -> Callable:
    """
    A decorator for OptimizationAlgorithmBase.solve method.
    Checks basic conditions.
    """

    @wraps(func)
    def solver_wrapper(
            self: Any,
            opt_prob: OptimizationProblem,
            initial_x_array_2d: Optional[ndarray] = None,
            initial_lambda_array_2d: Optional[ndarray] = None,
            initial_nu_array_2d: Optional[ndarray] = None,
            **kwargs
    ) -> OptimizationResult:
        logger.debug(self.__class__)
        logger.debug(opt_prob.__class__)
        logger.debug(initial_x_array_2d.__class__)
        logger.debug(initial_lambda_array_2d.__class__)
        logger.debug(initial_nu_array_2d.__class__)

        if initial_x_array_2d is None:
            initial_x_array_2d = randn(1, opt_prob.domain_dim)

        if opt_prob.num_ineq_cnst > 0 and initial_lambda_array_2d is None:
            initial_lambda_array_2d = randn(1, opt_prob.num_ineq_cnst)

        if opt_prob.num_eq_cnst > 0 and initial_nu_array_2d is None:
            initial_nu_array_2d = randn(1, opt_prob.num_eq_cnst)

        assert initial_lambda_array_2d is None or initial_x_array_2d.shape[0] == initial_lambda_array_2d.shape[0]
        assert initial_nu_array_2d is None or initial_x_array_2d.shape[0] == initial_nu_array_2d.shape[0]

        assert initial_x_array_2d.shape[1] == opt_prob.domain_dim
        assert initial_lambda_array_2d is None or initial_lambda_array_2d.shape[1] == opt_prob.num_ineq_cnst
        assert initial_nu_array_2d is None or initial_nu_array_2d.shape[1] == opt_prob.num_eq_cnst

        return func(self, opt_prob, initial_x_array_2d, initial_lambda_array_2d, initial_nu_array_2d, **kwargs)

    return solver_wrapper


def convex_solver(func: Callable) -> Callable:
    """
    A decorator for OptimizationAlgorithmBase.solve method.
    Checks whether an optimization problem has an equality constraint and it is linear.
    """

    @wraps(func)
    def convex_solver_wrapper(self: Any, opt_prob: OptimizationProblem, *args, **kwargs) -> OptimizationResult:
        assert opt_prob.is_convex

        return func(self, opt_prob, *args, **kwargs)

    return convex_solver_wrapper


def single_obj_solver(func: Callable) -> Callable:
    """
    A decorator for OptimizationAlgorithmBase.solve method.
    Checks whether an optimization problem is a single objective optimization problem.
    """

    @wraps(func)
    def single_obj_solver_wrapper(self: Any, opt_prob: OptimizationProblem, *args, **kwargs) -> OptimizationResult:
        assert opt_prob.obj_fcn is None or opt_prob.obj_fcn.num_outputs == 1
        return func(self, opt_prob, *args, **kwargs)

    return single_obj_solver_wrapper


def eq_cnst_solver(func: Callable) -> Callable:
    """
    A decorator for OptimizationAlgorithmBase.solve method.
    Checks whether an optimization problem is an equality constrained optimization problem.
    """

    @wraps(func)
    def eq_cnst_solver_wrapper(self: Any, opt_prob: OptimizationProblem, *args, **kwargs) -> OptimizationResult:
        assert opt_prob.obj_fcn is not None
        assert opt_prob.ineq_cnst_fcn is None
        assert opt_prob.eq_cnst_fcn is not None

        return func(self, opt_prob, *args, **kwargs)

    return eq_cnst_solver_wrapper


def linear_eq_cnst_solver(func: Callable) -> Callable:
    """
    A decorator for OptimizationAlgorithmBase.solve method.
    Checks whether an optimization problem has an equality constraint and it is linear.
    """

    @wraps(func)
    def linear_eq_cnst_solver_wrapper(self: Any, opt_prob: OptimizationProblem, *args, **kwargs) -> OptimizationResult:
        assert opt_prob.eq_cnst_fcn is not None and opt_prob.eq_cnst_fcn.is_affine
        return func(self, opt_prob, *args, **kwargs)

    return linear_eq_cnst_solver_wrapper
