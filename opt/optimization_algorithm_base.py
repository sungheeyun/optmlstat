from typing import Optional, List
from abc import ABC, abstractmethod

from numpy import ndarray

from opt.opt_prob.optimization_problem import OptimizationProblem
from opt.optimization_result import OptimizationResult


class OptimizationAlgorithmBase(ABC):
    """
    Optimization Algorithm
    """

    @abstractmethod
    def solve(
        self, opt_prob: OptimizationProblem, initial_point_or_list: Optional[ndarray, List[ndarray]]
    ) -> OptimizationResult:
        """
        Solve the optimization problem.

        Parameters
        ----------
        opt_prob:
         OptimizationProblem instance
        initial_point_or_list:
         If is ndarray, it is the initial point of a sequential method.
         If is list of ndarray, it is the initial points of decentralized or distributed algorithm
         or evolutionary algorithm.

        Returns
        -------
        optimization_result:
         OptimizationResult instance
        """
        pass
