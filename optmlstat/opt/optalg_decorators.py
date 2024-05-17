"""
decorators for optimization
mostly for checking requirements for each solver
"""

from functools import wraps
from logging import Logger, getLogger
from typing import Callable

import numpy as np

from optmlstat.functions.basic_functions.affine_function import AffineFunction
from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_res import OptResults
from optmlstat.opt.optalgs.optalg_base import OptAlgBase

logger: Logger = getLogger()


# DONE (H) Sridhar told me that a decorator shouldn't add any functionalities
#  just checking conditions. probably the below decorator violates that condition.
#  Review whether the below decorators satisfy the requirements after a little research on this
#  aspect.
#  done on 14-May-2024 - now decorators do check things only


def solver(func: Callable) -> Callable:
    """
    A decorator for OptimizationAlgorithmBase.solve method.
    Checks basic conditions.
    """

    @wraps(func)
    def solver_wrapper(
        self: OptAlgBase,
        opt_prob: OptProb,
        opt_param: OptParams,
        verbose: bool,
        /,
        *,
        initial_x_array_2d: np.ndarray,
        initial_lambda_array_2d: np.ndarray | None = None,
        initial_nu_array_2d: np.ndarray | None = None,
        **kwargs,
    ) -> OptResults:
        logger.debug(self.__class__)
        logger.debug(opt_prob.__class__)
        logger.debug(initial_x_array_2d.__class__)
        logger.debug(initial_lambda_array_2d.__class__)
        logger.debug(initial_nu_array_2d.__class__)

        # if initial_x_array_2d is None:
        #     initial_x_array_2d = randn(1, opt_prob.dim_domain)
        #
        # if opt_prob.num_ineq_cnst > 0 and initial_lambda_array_2d is None:
        #     initial_lambda_array_2d = randn(1, opt_prob.num_ineq_cnst)
        #
        # if opt_prob.num_eq_cnst > 0 and initial_nu_array_2d is None:
        #     initial_nu_array_2d = randn(1, opt_prob.num_eq_cnst)

        assert (
            initial_lambda_array_2d is None
            or initial_x_array_2d.shape[0] == initial_lambda_array_2d.shape[0]
        )
        assert (
            initial_nu_array_2d is None
            or initial_x_array_2d.shape[0] == initial_nu_array_2d.shape[0]
        )

        assert initial_x_array_2d.shape[1] == opt_prob.dim_domain
        assert (
            initial_lambda_array_2d is None
            or initial_lambda_array_2d.shape[1] == opt_prob.num_ineq_cnst
        )
        assert initial_nu_array_2d is None or initial_nu_array_2d.shape[1] == opt_prob.num_eq_cnst

        return func(
            self,
            opt_prob,
            opt_param,
            verbose,
            initial_x_array_2d=initial_x_array_2d,
            initial_lambda_array_2d=initial_lambda_array_2d,
            initial_nu_array_2d=initial_nu_array_2d,
            **kwargs,
        )

    return solver_wrapper


def convex_solver(func: Callable) -> Callable:
    """
    A decorator for OptimizationAlgorithmBase.solve method.
    Checks whether an OPTIMIZATION PROBLEM IS CONVEX
    """

    @wraps(func)
    def convex_solver_wrapper(
        self: OptAlgBase, opt_prob: OptProb, verbose: bool, *args, **kwargs
    ) -> OptResults:
        assert opt_prob.is_convex

        return func(self, opt_prob, verbose, *args, **kwargs)

    return convex_solver_wrapper


def single_obj_solver(func: Callable) -> Callable:
    """
    A decorator for OptimizationAlgorithmBase.solve method.
    Checks whether an optimization problem is a SINGLE OBJECTIVE OPTIMIZATION PROBLEM.
    """

    @wraps(func)
    def single_obj_solver_wrapper(
        self: OptAlgBase, opt_prob: OptProb, verbose: bool, *args, **kwargs
    ) -> OptResults:
        assert opt_prob.obj_fcn is None or opt_prob.obj_fcn.num_outputs == 1
        return func(self, opt_prob, verbose, *args, **kwargs)

    return single_obj_solver_wrapper


def eq_cnst_solver(func: Callable) -> Callable:
    """
    A decorator for OptimizationAlgorithmBase.solve method.
    checks whether an optimization problem has
    - OBJ FNC & EQ CONST ONLY
    """

    @wraps(func)
    def eq_cnst_solver_wrapper(
        self: OptAlgBase, opt_prob: OptProb, verbose: bool, *args, **kwargs
    ) -> OptResults:
        assert opt_prob.obj_fcn is not None
        assert opt_prob.eq_cnst_fcn is not None
        assert opt_prob.ineq_cnst_fcn is None

        return func(self, opt_prob, verbose, *args, **kwargs)

    return eq_cnst_solver_wrapper


def no_ineq_cnst_solver(func: Callable) -> Callable:
    """
    A decorator for OptimizationAlgorithmBase.solve method.
    Checks whether an optimization problem has
    NO INEQ CONSTRAINTS
    """

    @wraps(func)
    def wrapper(self: OptAlgBase, opt_prob: OptProb, verbose: bool, *args, **kwargs) -> OptResults:
        assert opt_prob.ineq_cnst_fcn is None

        return func(self, opt_prob, verbose, *args, **kwargs)

    return wrapper


def linear_eq_cnst_solver(func: Callable) -> Callable:
    """
    A decorator for OptimizationAlgorithmBase.solve method.
    Checks whether equality constraints has
    - EITHER NO EQ CONSTRAINT OR LINEAR EQ CONSTRAINTS IF IT HAS
    """

    @wraps(func)
    def wrapper(self: OptAlgBase, opt_prob: OptProb, verbose: bool, *args, **kwargs) -> OptResults:
        assert opt_prob.eq_cnst_fcn is None or isinstance(
            opt_prob.eq_cnst_fcn, AffineFunction
        ), opt_prob.eq_cnst_fcn
        return func(self, opt_prob, verbose, *args, **kwargs)

    return wrapper


def unconstrained_opt_solver(func: Callable) -> Callable:
    """
    check whether opt prob has no eq constrainta and ineq constraints
    doesn't care whether it has obj fcn
    """

    @wraps(func)
    def unconstrained_opt_solver_wrapper(
        self: OptAlgBase, opt_prob: OptProb, verbose: bool, *args, **kwargs
    ) -> OptResults:
        assert opt_prob.num_eq_cnst == 0 and opt_prob.num_ineq_cnst == 0, (
            opt_prob.num_eq_cnst,
            opt_prob.num_ineq_cnst,
        )
        return func(self, opt_prob, verbose, *args, **kwargs)

    return unconstrained_opt_solver_wrapper


def differentiable_obj_required_solver(func: Callable) -> Callable:
    """
    checks whether all obj fcn, eq & ineq constraint functions
    are differentiable
    """

    @wraps(func)
    def differentiable_obj_required_solver_wrapper(
        self: OptAlgBase, opt_prob: OptProb, verbose: bool, *args, **kwargs
    ) -> OptResults:
        assert opt_prob.obj_fcn is not None
        assert opt_prob.obj_fcn.is_differentiable
        return func(self, opt_prob, verbose, *args, **kwargs)

    return differentiable_obj_required_solver_wrapper


def twice_differentiable_obj_required_solver(func: Callable) -> Callable:
    """
    checks whether all obj fcn, eq & ineq constraint functions
    are twice-differentiable
    """

    @wraps(func)
    def twice_differentiable_obj_required_solver_wrapper(
        self: OptAlgBase, opt_prob: OptProb, verbose: bool, *args, **kwargs
    ) -> OptResults:
        assert opt_prob.obj_fcn is not None
        assert opt_prob.obj_fcn.is_twice_differentiable
        return func(self, opt_prob, verbose, *args, **kwargs)

    return twice_differentiable_obj_required_solver_wrapper
