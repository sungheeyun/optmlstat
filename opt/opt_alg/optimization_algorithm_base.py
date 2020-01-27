from abc import abstractmethod
from typing import Optional, List, Union, Any

from numpy import ndarray

from basic_modueles.class_base import OptMLStatClassBase
from opt.opt_prob import OptimizationProblem


class OptimizationAlgorithmBase(OptMLStatClassBase):
    """
    Optimization Algorithm
    """

    def __init__(self, learning_rate: float) -> None:
        self._learning_rate: float = learning_rate

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @abstractmethod
    def solve(
        self, opt_prob: OptimizationProblem, initial_point_or_list: Optional[Union[ndarray, List[ndarray]]]
    ) -> Any:
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
