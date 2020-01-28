from typing import Optional, Dict
from logging import Logger, getLogger

from basic_modueles.class_base import OptMLStatClassBase
from opt.opt_prob import OptimizationProblem
from opt.opt_alg.optimization_algorithm_base import OptimizationAlgorithmBase
from opt.iteration import Iteration
from opt.opt_iterate import OptimizationIterate
from opt.opt_prob_eval import OptimizationProblemEvaluation

logger: Logger = getLogger()


# TODO (5) implement plotter for OptimizaitonResult


class OptimizationResult(OptMLStatClassBase):
    """
    Stores optimization history and final results.
    """

    def __init__(self, opt_prob: OptimizationProblem, opt_alg: OptimizationAlgorithmBase) -> None:
        self._opt_prob: OptimizationProblem = opt_prob
        self._opt_alg: OptimizationAlgorithmBase = opt_alg
        self._iter_iterate_dict: Dict[Iteration, OptimizationIterate] = dict()

    def register_solution(
        self,
        iteration: Iteration,
        primal_prob_evaluation: OptimizationProblemEvaluation,
        dual_prob_evaluation: Optional[OptimizationProblemEvaluation] = None,
    ) -> None:
        assert iteration not in self._iter_iterate_dict

        self._iter_iterate_dict[iteration] = OptimizationIterate(primal_prob_evaluation, dual_prob_evaluation)

    @property
    def opt_prob(self) -> OptimizationProblem:
        return self._opt_prob

    @property
    def opt_alg(self) -> OptimizationAlgorithmBase:
        return self._opt_alg

    @property
    def iter_iterate_dict(self) -> Dict[Iteration, OptimizationIterate]:
        return self._iter_iterate_dict

    @property
    def final_iterate(self) -> OptimizationIterate:
        return sorted(self._iter_iterate_dict.items())[-1][1]
