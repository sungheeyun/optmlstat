from abc import abstractmethod
from typing import Any

from optmlstat.basic_modules.class_base import OMSClassBase
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_parameter import OptParams


class OptAlgBase(OMSClassBase):
    """
    Optimization Algorithm
    """

    @abstractmethod
    def solve(self, opt_prob: OptProb, opt_param: OptParams, *args, **kwargs) -> Any:
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
