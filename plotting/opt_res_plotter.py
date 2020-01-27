from typing import List, Tuple, Optional
from dataclasses import dataclass
from logging import Logger, getLogger

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from opt.iteration import Iteration
from opt.opt_iterate import OptimizationIterate
from opt.opt_res import OptimizationResult

logger: Logger = getLogger()


@dataclass(frozen=True)
class OptimizationResultPlotter:
    """
    Data class responsible for plotting optimization results
    """

    opt_res: OptimizationResult

    def plot_primal_and_dual_objs(self, ax: Axes, *args, **kwargs) -> Tuple[List[Line2D], List[Line2D]]:
        iteration_list: List[Iteration]
        opt_iterate_list: List[OptimizationIterate]
        iteration_list, opt_iterate_list = zip(*sorted(self.opt_res.iter_iterate_dict.items(), key=lambda x: x[0]))

        outer_iter_list: List[int] = [iteration.outer_iter for iteration in iteration_list]
        primal_obj_fcn_list: List[float] = [
            opt_iterate.primal_prob_evaluation.obj_fcn_array_2d[0, 0] for opt_iterate in opt_iterate_list
        ]
        dual_obj_fcn_list: List[Optional[float]] = [
            None
            if opt_iterate.dual_prob_evaluation is None
            else opt_iterate.dual_prob_evaluation.obj_fcn_array_2d[0, 0]
            for opt_iterate in opt_iterate_list
        ]

        res_1: List[Line2D] = ax.plot(
            outer_iter_list, primal_obj_fcn_list, label=r"$f(x^{(k)})$: primal obj", *args, **kwargs
        )
        res_2: List[Line2D] = ax.plot(
            outer_iter_list, dual_obj_fcn_list, label=r"$g(\lambda^{(k)})$: dual obj", *args, **kwargs
        )

        ax.legend()
        ax.set_xlabel("outer iteration")

        return res_1, res_2
