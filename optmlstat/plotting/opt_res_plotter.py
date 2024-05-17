"""
Optimization result plotter
"""

from dataclasses import dataclass
from logging import Logger, getLogger

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from pandas import DataFrame

from optmlstat.linalg.utils import get_random_orthogonal_array
from optmlstat.opt.iteration import Iteration
from optmlstat.opt.opt_iterate import OptimizationIterate
from optmlstat.opt.opt_res import OptResults
from optmlstat.plotting.plotter import plot_fcn_contour
from optmlstat.plotting.trajectory_obj_val_progress_animation import (
    TrajectoryObjValProgressAnimation,
)
from optmlstat.functions.exceptions import ValueUnknownException

logger: Logger = getLogger()


# CANCELED (2) implement a class containing the information re the plotting options
#  done on 14-May-2024 - decided not to do this


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
        self, ax: Axes, gap_axis: Axes | None, /, *args, **kwargs
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

        # iter_plot_fcn = ax.plot
        # iter_plot_fcn = axis.semilogy
        # gap_axis: Axes | None = kwargs.pop("gap_axis", None)
        # true_opt_val: float | None = kwargs.pop("true_opt_val", None)

        true_opt_val: float | None = None
        try:
            true_opt_val = self.opt_res.opt_prob.optimum_value  # type:ignore
        except ValueUnknownException:
            pass

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
            opt_iterate.dual_prob_evaluation.obj_fcn_array_2d.shape[0]
            for opt_iterate in opt_iterate_list
            if opt_iterate.dual_prob_evaluation is not None
            and opt_iterate.dual_prob_evaluation.obj_fcn_array_2d is not None
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

        primal_objs: list[Line2D] = ax.plot(
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
            primal_objs.extend(ax.plot(outer_iter, obj_fcn, *args, **kwargs))

            if gap_axis is not None and true_opt_val is not None:
                gap_axis.semilogy(outer_iter, np.array(obj_fcn) - true_opt_val, *args, **kwargs)

        if true_opt_val is not None:
            primal_objs.extend(ax.plot(ax.get_xlim(), np.ones(2) * true_opt_val, "r-", linewidth=2))

        if dual_obj_fcn_array_2d is not None:
            dual_objs = ax.plot(
                outer_iteration_list,
                dual_obj_fcn_array_2d[:, 0],
                label=r"$g(\lambda^{(k)})$: dual obj",
                *args,
                **kwargs,
            )
            assert dual_objs is not None
            dual_objs.extend(
                ax.plot(outer_iteration_list, dual_obj_fcn_array_2d[:, 1:], *args, **kwargs)
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

        for _ax in [ax, gap_axis]:
            if _ax is None:
                continue
            _ax.legend(fontsize=self.legend_font_size)
            _ax.set_xlabel("outer iteration", fontsize=self.xlabel_font_size)
            _ax.tick_params(
                axis="x",
                which="major",
                labelsize=self.major_xtick_label_font_size,
            )
            _ax.tick_params(
                axis="y",
                which="major",
                labelsize=self.major_ytick_label_font_size,
            )

        return primal_objs, dual_objs, dual_gap, primal_subopt, dual_subopt

    # CANCELED (2) add a method for drawing 3-d trajectories.
    #  cancelled on 12-May-2024
    # TODO (M) written in 2020
    #  add method for drawing dual variable trajectories

    # DONE (M)
    #  add a method for drawing variable trajectories and (primal and/or dual) obj functions
    #  together.
    #  done on 13-May-2024

    # CANCELED (M) add arguments for selection of variables to draw
    #  canceled on 14-May-2024 - decide not to do this because currently we do projection using two
    #  randomly chosen two orthonormal vectors when # primal variables is greater than two
    def animate_primal_sol(
        self,
        ax: Axes,
        iter_axes: list[Axes],
        /,
        *,
        head_ratio: float = 0.1,
        max_num_iterations_to_draw: int = 1000000,
        **kwargs,
    ) -> TrajectoryObjValProgressAnimation:
        """
        Create animation for primal solution trajectories.

        Parameters
        ----------
        ax:
         Axes
        iter_axes:
         Axes to iteration plots
        head_ratio:
         the ratio of the head part when drawing the trajectory
        max_num_iterations_to_draw:
         maximum number of iterations to draw

        Returns
        -------
        multi_axes_animation:
         TrajectoryObjValProgressAnimation:
        """

        # contour: bool = kwargs.pop("contour", False)
        # contour_xlim: tuple[float, float] = kwargs.pop("contour_xlim", (-1.0, 1.0))
        # contour_ylim: tuple[float, float] = kwargs.pop("contour_ylim", (-1.0, 1.0))

        assert 0.0 < head_ratio < 1.0

        proj_array_2d: np.ndarray = np.eye(2)  # projection matrix n-by-2
        assert self.opt_res.opt_prob.dim_domain > 1, self.opt_res.opt_prob.dim_domain
        if self.opt_res.opt_prob.dim_domain > 2:
            proj_array_2d = get_random_orthogonal_array(self.opt_res.opt_prob.dim_domain)[:, :2]

        opt_iterate_list: list[OptimizationIterate]
        _, opt_iterate_list = self.opt_res.iteration_iterate_list
        selected_opt_iterate_list: list[OptimizationIterate] = opt_iterate_list[
            :max_num_iterations_to_draw
        ]

        time_array_1d: np.ndarray = np.linspace(0.0, 1.0, len(selected_opt_iterate_list))

        x_array_3d: np.ndarray = np.array(  # {iter} x {data} x {0,1}
            [
                np.dot(opt_iterate.x_array_2d, proj_array_2d)
                for opt_iterate in selected_opt_iterate_list
            ]
        )

        x_array_2d: np.ndarray = x_array_3d[:, :, 0]
        y_array_2d: np.ndarray = x_array_3d[:, :, 1]

        opt_progress_animation: TrajectoryObjValProgressAnimation = (
            TrajectoryObjValProgressAnimation(
                ax.get_figure(),  # type:ignore
                [ax] * x_array_2d.shape[1],
                iter_axes,
                time_array_1d,
                x_array_2d,
                y_array_2d,
                head_time_period=head_ratio,
                **kwargs,
            )
        )

        assert self.opt_res.opt_prob.obj_fcn is not None
        optimum_point: np.ndarray | None = None
        try:
            optimum_point = self.opt_res.opt_prob.optimum_point
        except ValueUnknownException:
            pass
        ax.axis("equal")
        plot_fcn_contour(
            ax,
            self.opt_res.opt_prob.obj_fcn,
            proj_array_2d,
            center=optimum_point,
            xlim=ax.get_xlim(),
            ylim=ax.get_ylim(),
            levels=20,
        )
        if proj_array_2d.shape[0] > 2:
            ax.set_xlabel("v1")
            ax.set_ylabel("v2")
        else:
            ax.set_xlabel("[" + ", ".join([f"{x:g}" for x in proj_array_2d[:, 0]]) + "]")
            ax.set_ylabel("[" + ", ".join([f"{x:g}" for x in proj_array_2d[:, 1]]) + "]")

        plt.show()

        return opt_progress_animation
