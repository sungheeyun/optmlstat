from typing import List, Tuple, Optional
from dataclasses import dataclass
from logging import Logger, getLogger

from pandas import DataFrame
from numpy import ndarray, vstack, nan
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from opt.iteration import Iteration
from opt.opt_iterate import OptimizationIterate
from opt.opt_res import OptimizationResult

logger: Logger = getLogger()


# TODO (2) implement a class containing the information re the plotting options

@dataclass(frozen=True)
class OptimizationResultPlotter:
    """
    Data class responsible for plotting optimization results
    """

    opt_res: OptimizationResult
    legend_font_size: float = 15.0
    xlabel_font_size: float = 15.0
    major_xtick_label_font_size: float = 15.0
    major_ytick_label_font_size: float = 15.0

    def plot_primal_and_dual_objs(
        self, axis: Axes, *args, **kwargs
    ) -> Tuple[List[Line2D], Optional[List[Line2D]], Optional[List[Line2D]]]:
        gap_axis: Optional[Axes] = kwargs.pop("gap_axis", None)

        iteration_list: List[Iteration]
        opt_iterate_list: List[OptimizationIterate]
        iteration_list, opt_iterate_list = zip(*sorted(self.opt_res.iter_iterate_dict.items(), key=lambda x: x[0]))

        outer_iter_list: List[int] = [iteration.outer_iter for iteration in iteration_list]
        primal_obj_fcn_array_2d: ndarray = vstack(
            [opt_iterate.primal_prob_evaluation.obj_fcn_array_2d[:, 0] for opt_iterate in opt_iterate_list]
        )

        logger.info(primal_obj_fcn_array_2d[:3, :])

        dual_obj_fcn_dim_list: List[int] = [
            opt_iterate.dual_prob_evaluation.obj_fcn_array_2d.shape[0]
            for opt_iterate in opt_iterate_list
            if opt_iterate.dual_prob_evaluation is not None
        ]

        dual_obj_fcn_array_2d: Optional[ndarray] = None
        if len(dual_obj_fcn_dim_list) > 0:
            assert len(set(dual_obj_fcn_dim_list)) == 1
            dual_dim: int = dual_obj_fcn_dim_list[0]

            dual_obj_fcn_array_2d = vstack(
                [
                    [nan] * dual_dim
                    if opt_iterate.dual_prob_evaluation is None
                    or opt_iterate.dual_prob_evaluation.obj_fcn_array_2d is None
                    else opt_iterate.dual_prob_evaluation.obj_fcn_array_2d[:, 0]
                    for opt_iterate in opt_iterate_list
                ]
            )

        line2d_line_1: List[Line2D] = axis.plot(
            outer_iter_list[1:], primal_obj_fcn_array_2d[1:, 0], label=r"$f(x^{(k)})$: primal obj", *args, **kwargs
        )
        line2d_line_1.extend(axis.plot(outer_iter_list[1:], primal_obj_fcn_array_2d[1:, 1:], *args, **kwargs))

        line2d_line_2: Optional[List[Line2D]] = None
        if dual_obj_fcn_array_2d is not None:
            line2d_line_2 = axis.plot(
                outer_iter_list, dual_obj_fcn_array_2d[:, 0], label=r"$g(\lambda^{(k)})$: dual obj", *args, **kwargs
            )
            assert line2d_line_2 is not None
            line2d_line_2.extend(axis.plot(outer_iter_list, dual_obj_fcn_array_2d[:, 1:], *args, **kwargs))

        line2d_list_3: Optional[List[Line2D]] = None
        if gap_axis is not None and dual_obj_fcn_array_2d is not None:
            gap_array_2d: ndarray = (
                DataFrame(primal_obj_fcn_array_2d) - DataFrame(dual_obj_fcn_array_2d)
            ).abs().to_numpy()
            line2d_list_3 = gap_axis.semilogy(
                outer_iter_list,
                gap_array_2d[:, 0],
                label=r"$|f(x^{(k)})| - g(\lambda^{(k)})$: optimality certificate",
                *args,
                **kwargs
            )
            assert line2d_list_3 is not None
            line2d_list_3.extend(gap_axis.semilogy(outer_iter_list, gap_array_2d[:, 1:], *args, **kwargs))

        for ax in [axis, gap_axis]:
            if ax is None:
                continue
            ax.legend(fontsize=self.legend_font_size)
            ax.set_xlabel("outer iteration", fontsize=self.xlabel_font_size)
            ax.tick_params(axis="x", which="major", labelsize=self.major_xtick_label_font_size)
            ax.tick_params(axis="y", which="major", labelsize=self.major_ytick_label_font_size)

        return line2d_line_1, line2d_line_2, line2d_list_3
