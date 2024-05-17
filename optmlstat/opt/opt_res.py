from logging import Logger, getLogger

import numpy as np
from numpy import linalg
from typing import Any

from optmlstat.basic_modules.class_base import OMSClassBase
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.optalgs.optalg_base import OptAlgBase
from optmlstat.opt.iteration import Iteration
from optmlstat.opt.opt_iterate import OptimizationIterate
from optmlstat.opt.opt_prob_eval import OptProbEval
from optmlstat.functions.exceptions import ValueUnknownException

logger: Logger = getLogger()


class OptResults(OMSClassBase):
    """
    Stores optimization history and final results.
    """

    def __init__(self, opt_prob: OptProb, opt_alg: OptAlgBase) -> None:
        self._opt_prob: OptProb = opt_prob
        self._opt_alg: OptAlgBase = opt_alg
        self._iter_iterate_dict: dict[Iteration, OptimizationIterate] = dict()

        self._solve_time: float | None = None

    @property
    def solve_time(self) -> float | None:
        return self._solve_time

    @solve_time.setter
    def solve_time(self, value: float) -> None:
        self._solve_time = value

    def register_solution(
        self,
        iteration: Iteration,
        primal_prob_evaluation: OptProbEval,
        dual_prob_evaluation: OptProbEval,
        verbose: bool,
        /,
        *,
        terminated: np.ndarray | None = None,
        stopping_criteria_info: dict[str, Any] | None = None,
    ) -> None:
        _stopping_criteria_info: dict[str, Any] = (
            dict() if stopping_criteria_info is None else stopping_criteria_info
        )
        assert iteration not in self._iter_iterate_dict

        if terminated is None:
            terminated = np.array([False] * primal_prob_evaluation.x_array_2d.shape[0])

        self._iter_iterate_dict[iteration] = OptimizationIterate(
            primal_prob_evaluation, dual_prob_evaluation, terminated
        )

        logger.info(
            f"iter: {iteration.outer_iteration}/{iteration.inner_iteration}"
            f" - best: {self.best_obj_values.min()}"
            f" - avg. grad norm: {self.last_obj_grad_norm_avg.min()}"
        )
        for key in sorted(_stopping_criteria_info):
            logger.info(
                f"\t{key}: {self.opt_progress_report_info_to_str(_stopping_criteria_info[key])}"
            )
        logger.info(f"\tterminated: {self.opt_progress_report_info_to_str(terminated)}")

        if verbose:
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

    def result_analysis(self) -> None:
        num_iterations_list: list[int] = self.num_iterations_list

        logger.info("optimization result analysis")
        logger.info(f"\topt. prob.: {self.opt_prob}")
        logger.info(f"\t# members: {self.population_size}")
        logger.info(f"\toptimization time: {self.solve_time:.3g} sec.")

        logger.info(f"\t# iters: {num_iterations_list}")
        logger.info(f"\tavg # iters: {np.array(num_iterations_list).mean()}")
        logger.info(f"\tavg final x: {self.final_iterate.x_array_2d.mean(axis=0)}")
        logger.info(f"\tavg final lambda: {self.final_iterate.lambda_array_2d.mean(axis=0)}")
        logger.info(f"\tavg final nu: {self.final_iterate.nu_array_2d.mean(axis=0)}")
        logger.info(f"\tbest obj values: {self.best_obj_values}")

        try:
            true_opt_val: np.ndarray | float = self.opt_prob.optimum_value
            logger.info(f"\tabs suboptimality: {self.best_obj_values - true_opt_val}")
            logger.info(
                f"\trel suboptimality: {(self.best_obj_values - true_opt_val)/np.abs(true_opt_val)}"
            )
        except ValueUnknownException:
            pass

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
    def last_obj_grad_norm_avg(self) -> np.ndarray:
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

    @staticmethod
    def opt_progress_report_info_to_str(value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.ndim == 1:
            return list(value)
        return value
