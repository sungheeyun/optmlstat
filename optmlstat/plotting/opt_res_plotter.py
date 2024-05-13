"""
Optimization result plotter
"""

from dataclasses import dataclass
from logging import Logger, getLogger

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pandas import DataFrame

from optmlstat.opt.iteration import Iteration
from optmlstat.opt.opt_iterate import OptimizationIterate
from optmlstat.opt.opt_res import OptResults
from optmlstat.plotting.multi_axes_animation import MultiAxesAnimation
from optmlstat.plotting.plotter import plot_fcn_contour

logger: Logger = getLogger()


# TODO (2) implement a class containing the information re the plotting options


@dataclass(frozen=True)
class OptimizationResultPlotter:
    """
    Data class responsible for plotting optimization results
    """

    opt_res: OptResults
    legend_font_size: float = 10.0
    xlabel_font_size: float = 10.0
    major_xtick_label_font_size: float = 10.0
    major_ytick_label_font_size: float = 10.0

    def plot_primal_and_dual_objs(
        self, axis: Axes, gap_axis: Axes | None, true_opt_val: float | None, /, *args, **kwargs
    ) -> tuple[
        list[Line2D],
        list[Line2D] | None,
        list[Line2D] | None,
        list[Line2D] | None,
        list[Line2D] | None,
    ]:
        dual_objs: list[Line2D] | None = None
        dual_gap: list[Line2D] | None = None
        primal_subopt: list[Line2D] | None = None
        dual_subopt: list[Line2D] | None = None

        iter_plot_fcn = axis.plot
        # iter_plot_fcn = axis.semilogy
        # gap_axis: Axes | None = kwargs.pop("gap_axis", None)
        # true_opt_val: float | None = kwargs.pop("true_opt_val", None)

        iteration_list: list[Iteration]
        opt_iterate_list: list[OptimizationIterate]
        iteration_list, opt_iterate_list = self.opt_res.iteration_iterate_list

        outer_iteration_list: list[int] = Iteration.get_outer_iteration_list(iteration_list)

        num_pnts: int = self.opt_res.population_size
        primal_obj_fcn_array_2d: np.ndarray = self.opt_res.primal_1st_obj_fcn_iterates_2d

        plot_outer_iter: list[list[int]] = [list() for _ in range(num_pnts)]
        plot_obj_fcn: list[list[float]] = [list() for _ in range(num_pnts)]

        num_iterations_list: list[int] = self.opt_res.num_iterations_list
        for iter_idx, opt_iterate in enumerate(opt_iterate_list):
            iteration = iteration_list[iter_idx]

            for x_idx, num_iterations in enumerate(num_iterations_list):
                if iter_idx < num_iterations:
                    assert opt_iterate.primal_prob_evaluation.obj_fcn_array_2d is not None

                    plot_outer_iter[x_idx].append(iteration.outer_iteration)
                    plot_obj_fcn[x_idx].append(
                        opt_iterate.primal_prob_evaluation.obj_fcn_array_2d[x_idx, 0]
                    )

        dual_obj_fcn_dim_list: list[int] = [
            opt_iterate.dual_prob_evaluation.obj_fcn_array_2d.shape[0]  # type:ignore
            for opt_iterate in opt_iterate_list
            if opt_iterate.dual_prob_evaluation is not None
        ]

        dual_obj_fcn_array_2d: np.ndarray | None = None
        if len(dual_obj_fcn_dim_list) > 0:
            assert len(set(dual_obj_fcn_dim_list)) == 1
            dual_dim: int = dual_obj_fcn_dim_list[0]

            dual_obj_fcn_array_2d = np.vstack(
                [
                    (
                        [np.nan] * dual_dim
                        if opt_iterate.dual_prob_evaluation is None
                        or opt_iterate.dual_prob_evaluation.obj_fcn_array_2d is None
                        else opt_iterate.dual_prob_evaluation.obj_fcn_array_2d[:, 0]
                    )
                    for opt_iterate in opt_iterate_list
                ]
            )

        primal_objs: list[Line2D] = iter_plot_fcn(
            plot_outer_iter[0],
            plot_obj_fcn[0],
            label=r"$f(x^{(k)})$: primal obj",
            *args,
            **kwargs,
        )

        if gap_axis is not None and true_opt_val is not None:
            primal_subopt = gap_axis.semilogy(
                plot_outer_iter[0],
                np.array(plot_obj_fcn[0]) - true_opt_val,
                label=r"$f(x^{(k)}) - p^\ast$: primal suboptimality",
                *args,
                **kwargs,
            )

        for idx in range(1, len(plot_outer_iter)):
            outer_iter: list[int] = plot_outer_iter[idx]
            obj_fcn: list[float] = plot_obj_fcn[idx]
            primal_objs.extend(iter_plot_fcn(outer_iter, obj_fcn, *args, **kwargs))

            if gap_axis is not None and true_opt_val is not None:
                gap_axis.semilogy(outer_iter, np.array(obj_fcn) - true_opt_val, *args, **kwargs)

        if true_opt_val is not None:
            primal_objs.extend(
                axis.plot(axis.get_xlim(), np.ones(2) * true_opt_val, "r-", linewidth=2)
            )

        if dual_obj_fcn_array_2d is not None:
            dual_objs = iter_plot_fcn(
                outer_iteration_list,
                dual_obj_fcn_array_2d[:, 0],
                label=r"$g(\lambda^{(k)})$: dual obj",
                *args,
                **kwargs,
            )
            assert dual_objs is not None
            dual_objs.extend(
                iter_plot_fcn(outer_iteration_list, dual_obj_fcn_array_2d[:, 1:], *args, **kwargs)
            )

        if gap_axis is not None and dual_obj_fcn_array_2d is not None:
            gap_array_2d: np.ndarray = (
                (DataFrame(primal_obj_fcn_array_2d) - DataFrame(dual_obj_fcn_array_2d))
                .abs()
                .to_numpy()
            )
            dual_gap = gap_axis.semilogy(
                outer_iteration_list,
                gap_array_2d[:, 0],
                label=r"$|f(x^{(k)})| - g(\lambda^{(k)})$:" " optimality certificate",
                *args,
                **kwargs,
            )
            assert dual_gap is not None
            dual_gap.extend(
                gap_axis.semilogy(outer_iteration_list, gap_array_2d[:, 1:], *args, **kwargs)
            )

        for ax in [axis, gap_axis]:
            if ax is None:
                continue
            ax.legend(fontsize=self.legend_font_size)
            ax.set_xlabel("outer iteration", fontsize=self.xlabel_font_size)
            ax.tick_params(
                axis="x",
                which="major",
                labelsize=self.major_xtick_label_font_size,
            )
            ax.tick_params(
                axis="y",
                which="major",
                labelsize=self.major_ytick_label_font_size,
            )

        return primal_objs, dual_objs, dual_gap, primal_subopt, dual_subopt

    # TODO (2) add a method for drawing 3-d trajectories.
    #  CANCELLED on 12-May-2024
    # TODO (4) add method for drawing dual variable trajectories
    # TODO (3) add a method for drawing variable trajectories and
    #  (primal and/or dual) obj functions together.

    # TODO (4) add arguments for selection of variables to draw
    def animate_primal_sol(
        self, head_ratio: float = 0.1, max_num_iterations_to_draw: int = 1000000, **kwargs
    ) -> MultiAxesAnimation:
        """
        Create animation for primal solution trajectories.

        Parameters
        ----------
        head_ratio:
         the ratio of the head part when drawing the trajectory
        max_num_iterations_to_draw:
         maximum number of iterations to draw

        Returns
        -------
        multi_axes_animation:
         MultiAxesAnimation:
        """

        contour: bool = kwargs.pop("contour", False)
        contour_xlim: tuple[float, float] = kwargs.pop("contour_xlim", (-1.0, 1.0))
        contour_ylim: tuple[float, float] = kwargs.pop("contour_ylim", (-1.0, 1.0))

        assert 0.0 < head_ratio < 1.0

        opt_iterate_list: list[OptimizationIterate]
        _, opt_iterate_list = self.opt_res.iteration_iterate_list
        selected_opt_iterate_list: list[OptimizationIterate] = opt_iterate_list[
            :max_num_iterations_to_draw
        ]

        time_array_1d: np.ndarray = np.linspace(0.0, 1.0, len(selected_opt_iterate_list))

        idx1 = 0
        idx2 = 1

        x_array_2d: np.ndarray = np.vstack(
            [opt_iterate.x_array_2d[:, idx1] for opt_iterate in selected_opt_iterate_list]
        )
        y_array_2d: np.ndarray = np.vstack(
            [opt_iterate.x_array_2d[:, idx2] for opt_iterate in selected_opt_iterate_list]
        )

        # logger.info(time_array_1d.shape)
        # logger.info(x_array_2d.shape)
        # logger.info(y_array_2d.shape)

        figure: Figure
        axis: Axes

        from freq_used.plotting import get_figure

        figure = get_figure(1, 1, axis_width=5.0, axis_height=5.0)
        axis = figure.get_axes()[0]

        multi_axes_animation: MultiAxesAnimation = MultiAxesAnimation(
            figure,
            [axis] * x_array_2d.shape[1],
            time_array_1d,
            x_array_2d,
            y_array_2d,
            head_time_period=head_ratio,
            **kwargs,
        )

        if contour:
            assert self.opt_res.opt_prob.obj_fcn is not None
            plot_fcn_contour(
                axis,
                self.opt_res.opt_prob.obj_fcn,
                levels=30,
                xlim=contour_xlim,
                ylim=contour_ylim,
            )
            axis.set_xlim(contour_xlim)
            axis.set_ylim(contour_ylim)
            axis.axis("equal")

        plt.show()

        return multi_axes_animation
