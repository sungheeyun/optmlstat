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
