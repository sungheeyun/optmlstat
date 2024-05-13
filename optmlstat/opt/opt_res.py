from logging import Logger, getLogger

import numpy as np
from numpy import linalg

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
        iteration: Iteration,
        primal_prob_evaluation: OptProbEval,
        verbose: bool,
        /,
        *,
        dual_prob_evaluation: OptProbEval | None = None,
        terminated: np.ndarray | None = None,
    ) -> None:
        assert iteration not in self._iter_iterate_dict

        if terminated is None:
            terminated = np.array([False] * primal_prob_evaluation.x_array_2d.shape[0])

        self._iter_iterate_dict[iteration] = OptimizationIterate(
            primal_prob_evaluation, terminated, dual_prob_evaluation
        )

        logger.info(
            f"iter: {iteration.outer_iteration}/{iteration.inner_iteration}"
            f" - best: {self.best_obj_values.min()}"
            f" - avg. jac: {self.last_obj_jac_norm_avg.min()}"
        )
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

    def result_analysis(self, true_opt_val: np.ndarray | float | None = None) -> None:
        if isinstance(true_opt_val, float):
            true_opt_val = np.array([true_opt_val])

        num_iterations_list: list[int] = self.num_iterations_list

        logger.info("optimization result analysis")
        logger.info(f"\topt. prob.: {str(self.opt_prob)}")
        logger.info(f"\tobj fcn: {str(self.opt_prob.obj_fcn)}")
        logger.info(f"\tpopulation size: {str(self.population_size)}")

        logger.info(f"\t# iters: {num_iterations_list}")
        logger.info(f"\tavg # iters: {np.array(num_iterations_list).mean()}")
        logger.info(f"\tbest obj values: {self.best_obj_values}")

        if true_opt_val is not None:
            assert isinstance(true_opt_val, np.ndarray), true_opt_val.__class__
            assert true_opt_val.ndim == 1, true_opt_val.shape
            assert true_opt_val.size == self.best_obj_values.size

            logger.info(f"\tabs suboptimality: {self.best_obj_values - true_opt_val}")
            logger.info(
                f"\trel suboptimality: {(self.best_obj_values - true_opt_val)/np.abs(true_opt_val)}"
            )

    @property
    def iteration_iterate_list(self) -> tuple[list[Iteration], list[OptimizationIterate]]:
        _iteration_list, _opt_iterate_list = zip(*sorted(self.iter_iterate_dict.items()))
        return list(_iteration_list), list(_opt_iterate_list)

    @property
    def num_iterations_list(self) -> list[int]:
        return list(
            (
                ~np.vstack(
                    [opt_iterate.terminated for opt_iterate in self.iteration_iterate_list[1]]
                )
            ).sum(axis=0)
            + 1
        )

    @property
    def primal_1st_obj_fcn_iterates_2d(self) -> np.ndarray:
        return np.vstack(
            [
                opt_iterate.primal_prob_evaluation.obj_fcn_array_2d[:, 0]  # type:ignore
                for opt_iterate in self.iteration_iterate_list[1]
            ]
        )

    @property
    def population_size(self) -> int:
        return list(self._iter_iterate_dict.values())[0].primal_prob_evaluation.x_array_2d.shape[0]

    @property
    def best_obj_values(self) -> np.ndarray:
        return np.vstack(
            [
                iterate.primal_prob_evaluation.obj_fcn_array_2d  # type:ignore
                for iterate in self.iteration_iterate_list[1]
            ]
        ).min(axis=0)

    @property
    def last_obj_jac_norm_avg(self) -> np.ndarray:
        assert self.iteration_iterate_list[1][-1].primal_prob_evaluation.obj_fcn_jac_3d is not None
        assert (
            self.iteration_iterate_list[1][-1].primal_prob_evaluation.obj_fcn_jac_3d.ndim == 3
        ), self.iteration_iterate_list[1][-1].primal_prob_evaluation.obj_fcn_jac_3d.shape

        return np.array(
            [
                np.array([linalg.norm(jac_1d) for jac_1d in jac_2d]).mean()
                for jac_2d in self.iteration_iterate_list[1][
                    -1
                ].primal_prob_evaluation.obj_fcn_jac_3d
            ]
        ).mean()
