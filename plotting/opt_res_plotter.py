from typing import List, Tuple, Optional
from dataclasses import dataclass
from logging import Logger, getLogger

from pandas import DataFrame
from numpy import ndarray, vstack, nan, linspace
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from opt.iteration import Iteration
from opt.opt_iterate import OptimizationIterate
from opt.opt_res import OptimizationResult
from plotting.multi_axes_animation import MultiAxesAnimation

logger: Logger = getLogger()


# TODO (2) implement a class containing the information re the plotting options


@dataclass(frozen=True)
class OptimizationResultPlotter:
    """
    Data class responsible for plotting optimization results
    """

    opt_res: OptimizationResult
    legend_font_size: float = 10.0
    xlabel_font_size: float = 10.0
    major_xtick_label_font_size: float = 10.0
    major_ytick_label_font_size: float = 10.0

    def get_sorted_iteration_and_iterate(self) -> Tuple[List[Iteration], List[OptimizationIterate]]:
        iteration_list: List[Iteration]
        opt_iterate_list: List[OptimizationIterate]
        iteration_list, opt_iterate_list = zip(*sorted(self.opt_res.iter_iterate_dict.items()))
        return iteration_list, opt_iterate_list

    def plot_primal_and_dual_objs(
        self, axis: Axes, *args, **kwargs
    ) -> Tuple[List[Line2D], Optional[List[Line2D]], Optional[List[Line2D]]]:
        gap_axis: Optional[Axes] = kwargs.pop("gap_axis", None)

        iteration_list: List[Iteration]
        opt_iterate_list: List[OptimizationIterate]
        iteration_list, opt_iterate_list = self.get_sorted_iteration_and_iterate()
        outer_iteration_list: List[int] = Iteration.get_outer_iteration_list(iteration_list)

        primal_obj_fcn_array_2d: ndarray = vstack(
            [opt_iterate.primal_prob_evaluation.obj_fcn_array_2d[:, 0] for opt_iterate in opt_iterate_list]
        )

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
            outer_iteration_list[1:], primal_obj_fcn_array_2d[1:, 0], label=r"$f(x^{(k)})$: primal obj", *args, **kwargs
        )
        line2d_line_1.extend(axis.plot(outer_iteration_list[1:], primal_obj_fcn_array_2d[1:, 1:], *args, **kwargs))

        line2d_line_2: Optional[List[Line2D]] = None
        if dual_obj_fcn_array_2d is not None:
            line2d_line_2 = axis.plot(
                outer_iteration_list,
                dual_obj_fcn_array_2d[:, 0],
                label=r"$g(\lambda^{(k)})$: dual obj",
                *args,
                **kwargs
            )
            assert line2d_line_2 is not None
            line2d_line_2.extend(axis.plot(outer_iteration_list, dual_obj_fcn_array_2d[:, 1:], *args, **kwargs))

        line2d_list_3: Optional[List[Line2D]] = None
        if gap_axis is not None and dual_obj_fcn_array_2d is not None:
            gap_array_2d: ndarray = (
                DataFrame(primal_obj_fcn_array_2d) - DataFrame(dual_obj_fcn_array_2d)
            ).abs().to_numpy()
            line2d_list_3 = gap_axis.semilogy(
                outer_iteration_list,
                gap_array_2d[:, 0],
                label=r"$|f(x^{(k)})| - g(\lambda^{(k)})$: optimality certificate",
                *args,
                **kwargs
            )
            assert line2d_list_3 is not None
            line2d_list_3.extend(gap_axis.semilogy(outer_iteration_list, gap_array_2d[:, 1:], *args, **kwargs))

        for ax in [axis, gap_axis]:
            if ax is None:
                continue
            ax.legend(fontsize=self.legend_font_size)
            ax.set_xlabel("outer iteration", fontsize=self.xlabel_font_size)
            ax.tick_params(axis="x", which="major", labelsize=self.major_xtick_label_font_size)
            ax.tick_params(axis="y", which="major", labelsize=self.major_ytick_label_font_size)

        return line2d_line_1, line2d_line_2, line2d_list_3

    def animate_primal_sol(self) -> MultiAxesAnimation:

        opt_iterate_list: List[OptimizationIterate]
        _, opt_iterate_list = self.get_sorted_iteration_and_iterate()

        time_array_1d: ndarray = linspace(0.0, 2.0, len(opt_iterate_list))

        idx1 = 0
        idx2 = 1

        x_array_2d: ndarray = vstack([opt_iterate.x_array_2d[:, idx1] for opt_iterate in opt_iterate_list])
        y_array_2d: ndarray = vstack([opt_iterate.x_array_2d[:, idx2] for opt_iterate in opt_iterate_list])

        logger.info(time_array_1d.shape)
        logger.info(x_array_2d.shape)
        logger.info(y_array_2d.shape)

        figure: Figure
        axis: Axes

        figure, axis = plt.subplots()

        multi_axes_animation: MultiAxesAnimation = MultiAxesAnimation(
            figure,
            [axis] * x_array_2d.shape[1],
            time_array_1d,
            x_array_2d,
            y_array_2d,
        )

        axis.set_xlim(-5.0, 5.0)
        axis.set_ylim(-5.0, 5.0)

        return multi_axes_animation
