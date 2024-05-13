from logging import Logger, getLogger

import numpy as np

from optmlstat.basic_modules.class_base import OMSClassBase
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.optalgs.optalg_base import OptAlgBase
from optmlstat.opt.iteration import Iteration
from optmlstat.opt.opt_iterate import OptimizationIterate
from optmlstat.opt.opt_prob_eval import OptProbEval

logger: Logger = getLogger()


class OptResults(OMSClassBase):
    """
    Stores optimization history and final results.
    """

    def __init__(self, opt_prob: OptProb, opt_alg: OptAlgBase) -> None:
        self._opt_prob: OptProb = opt_prob
        self._opt_alg: OptAlgBase = opt_alg
        self._iter_iterate_dict: dict[Iteration, OptimizationIterate] = dict()

    def register_solution(
        self,
        *,
        iteration: Iteration,
        primal_prob_evaluation: OptProbEval,
        dual_prob_evaluation: OptProbEval | None = None,
        terminated: np.ndarray | None = None,
        verbose: bool = False,
    ) -> None:
        assert iteration not in self._iter_iterate_dict

        if terminated is None:
            terminated = np.array([False] * primal_prob_evaluation.x_array_2d.shape[0])

        self._iter_iterate_dict[iteration] = OptimizationIterate(
            primal_prob_evaluation, terminated, dual_prob_evaluation
        )

        logger.info(f"iter: {iteration.outer_iteration}/{iteration.inner_iteration}")
        if verbose:
            logger.info(f"\tterminated: {terminated}")
            logger.info(f"\tprimal: {primal_prob_evaluation}")
            logger.info(f"\tdual: {dual_prob_evaluation}")

    @property
    def opt_prob(self) -> OptProb:
        return self._opt_prob

    @property
    def opt_alg(self) -> OptAlgBase:
        return self._opt_alg

    @property
    def iter_iterate_dict(self) -> dict[Iteration, OptimizationIterate]:
        return self._iter_iterate_dict

    @property
    def final_iterate(self) -> OptimizationIterate:
        return sorted(self._iter_iterate_dict.items())[-1][1]
