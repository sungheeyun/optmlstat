from logging import Logger, getLogger
from typing import Any

import numpy as np
from numpy import linalg

from optmlstat.basic_modules.class_base import OMSClassBase
from optmlstat.functions.exceptions import ValueUnknownException, InfiniteNumberOfSolutionsException
from optmlstat.opt.iteration import Iteration
from optmlstat.opt.opt_iterate import OptimizationIterate
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_prob_eval import OptProbEval
from optmlstat.opt.optalgs.optalg_base import OptAlgBase

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

        self._x_diff_norm: float = np.inf
        self._lambda_diff_norm: float = np.inf
        self._nu_diff_norm: float = np.inf

    @property
    def solve_time(self) -> float | None:
        return self._solve_time

    @solve_time.setter
    def solve_time(self, value: float) -> None:
        self._solve_time = value

    @property
    def x_diff_norm(self) -> float:
        return self._x_diff_norm

    @property
    def lambda_diff_norm(self) -> float:
        return self._lambda_diff_norm

    @property
    def nu_diff_norm(self) -> float:
        return self._nu_diff_norm

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
        stopping_criteria_name: str | None = None,
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

        logger.info(f"iter: {iteration.outer_iteration}/{iteration.inner_iteration}")
        logger.info(f" - obj fcn: {self.last_obj_values}")
        logger.info(f" - avg. grad norm: {self.last_obj_grad_norm_avg.min()}")
        for key in sorted(_stopping_criteria_info):
            logger.info(f"\t{key}: {self.pretty_data_format(_stopping_criteria_info[key])}")
        logger.info(f"\tterminated: {self.pretty_data_format(terminated)}")

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
        logger.info(
            "\tavg final x: "
            + str(self.pretty_data_format(self.final_iterate.x_array_2d.mean(axis=0)))
        )
        try:
            logger.info(f"\t\toptimum x: {self.pretty_data_format(self.opt_prob.optimum_point)}")
            self._x_diff_norm = float(
                linalg.norm(self.opt_prob.optimum_point - self.final_iterate.x_array_2d)
            )
            logger.info(f"\t\tdiff x norm: {self.x_diff_norm}")
        except ValueUnknownException:
            pass

        logger.info(
            "\tavg final lambda: "
            f"{self.pretty_data_format(self.final_iterate.lambda_array_2d.mean(axis=0))}"
        )
        try:
            logger.info(
                f"\t\toptimum lambda: {self.pretty_data_format(self.opt_prob.optimum_lambda)}"
            )
            self._lambda_diff_norm = float(
                linalg.norm(self.opt_prob.optimum_lambda - self.final_iterate.lambda_array_2d)
            )
            logger.info(f"\t\tdiff lambda norm: {self.lambda_diff_norm}")
        except ValueUnknownException:
            pass

        logger.info(
            "\tavg final nu:"
            f" {self.pretty_data_format(self.final_iterate.nu_array_2d.mean(axis=0))}"
        )
        try:
            logger.info(f"\t\toptimum nu: {self.pretty_data_format(self.opt_prob.optimum_nu)}")
            self._nu_diff_norm = float(
                linalg.norm(self.opt_prob.optimum_nu - self.final_iterate.nu_array_2d)
            )
            logger.info(f"\t\tdiff nu norm: {self.nu_diff_norm}")
        except ValueUnknownException:
            pass

        logger.info(f"\tfinal obj values: {self.last_obj_values}")

        # logger.info(self.final_iterate.x_2d.mean(axis=0) - self.opt_prob.optimum_point)

        try:
            true_opt_val: np.ndarray | float = self.opt_prob.optimum_value
            logger.info(f"\t\toptimum value: {true_opt_val}")
            logger.info(f"\tabs suboptimality: {self.last_obj_values - true_opt_val}")
            logger.info(
                "\trel suboptimality: "
                f"{(self.last_obj_values - true_opt_val) / np.abs(true_opt_val)}"
            )
        except ValueUnknownException:
            pass

        try:
            logger.info(f"\ttrue dual optimum value: {-self.opt_prob.dual_problem.optimum_value}")
            logger.info(
                "\ttrue dual optimum point: "
                + f"{self.pretty_data_format(self.opt_prob.dual_problem.optimum_point)}"
            )
        except (ValueUnknownException, InfiniteNumberOfSolutionsException):
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
    def population_size(self) -> int:
        return list(self._iter_iterate_dict.values())[0].primal_prob_evaluation.x_array_2d.shape[0]

    @property
    def last_obj_values(self) -> np.ndarray:
        assert (
            self.iteration_iterate_list[1][-1].primal_prob_evaluation.obj_fcn_array_2d is not None
        )
        return self.iteration_iterate_list[1][-1].primal_prob_evaluation.obj_fcn_array_2d.mean(
            axis=0
        )

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
    def pretty_data_format(value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.ndim == 1:
            return list(value)
        return value

    @property
    def num_members(self) -> int:
        return list(self.iter_iterate_dict.values())[0].x_array_2d.shape[0]

    @property
    def primal_dual_plot_lists(
        self,
    ) -> tuple[list[list[int]], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        :return: list of iter lists, list of primal obj list
        """
        iter_list_list: list[list[int]] = list()
        primal_obj_list_list: list[np.ndarray] = list()
        dual_obj_list_list: list[np.ndarray] = list()
        primal_eq_list_list: list[np.ndarray] = list()

        _, iterate_list = self.iteration_iterate_list

        for member_idx, num_iterations in enumerate(self.num_iterations_list):
            iter_list_list.append(list(range(num_iterations)))
            primal_obj_list_list.append(
                np.vstack(
                    [
                        iterate_list[  # type:ignore
                            min(_iter, len(iterate_list) - 1)
                        ].primal_prob_evaluation.obj_fcn_array_2d[member_idx]
                        for _iter in range(num_iterations)
                    ]
                )
            )
            dual_obj_list_list.append(
                np.vstack(
                    [
                        iterate_list[  # type:ignore
                            min(_iter, len(iterate_list) - 1)
                        ].dual_prob_evaluation.obj_fcn_array_2d[member_idx]
                        for _iter in range(num_iterations)
                    ]
                )
            )
            primal_eq_list_list.append(
                np.vstack(
                    [
                        (
                            None  # type:ignore
                            if iterate_list[
                                min(_iter, len(iterate_list) - 1)
                            ].primal_prob_evaluation.eq_cnst_array_2d
                            is None
                            else iterate_list[  # type:ignore
                                min(_iter, len(iterate_list) - 1)
                            ].primal_prob_evaluation.eq_cnst_array_2d[member_idx]
                        )
                        for _iter in range(num_iterations)
                    ]
                )
            )

        return iter_list_list, primal_obj_list_list, dual_obj_list_list, primal_eq_list_list
