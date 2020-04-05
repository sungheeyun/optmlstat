from abc import abstractmethod
from typing import Any

from optmlstat.basic_modules.class_base import OptMLStatClassBase
from optmlstat.opt.opt_prob import OptimizationProblem
from optmlstat.opt.opt_parameter import OptimizationParameter


class OptimizationAlgorithmBase(OptMLStatClassBase):
    """
    Optimization Algorithm
    """

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @abstractmethod
    def solve(self, opt_prob: OptimizationProblem, opt_param: OptimizationParameter, *args, **kwargs) -> Any:
        """
        Solve the optimization problem.

        Parameters
        ----------
        opt_param
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
