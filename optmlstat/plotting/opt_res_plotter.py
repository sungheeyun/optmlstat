"""
Optimization result plotter
"""

from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from optmlstat.functions.exceptions import ValueUnknownException
from optmlstat.linalg.utils import get_random_orthogonal_array
from optmlstat.opt.opt_iterate import OptimizationIterate
from optmlstat.opt.opt_res import OptResults
from optmlstat.plotting.plotter import plot_fcn_contour
from optmlstat.plotting.trajectory_obj_val_progress_animation import (
    TrajectoryObjValProgressAnimation,
)

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
        self,
        obj_val_axis: Axes,
        primal_gap_axis: Axes | None,
        dual_gap_axis: Axes | None,
        /,
        *args,
        **kwargs,
    ) -> None:
        iter_list_list, primal_obj_list_list, dual_objs_list_list = (
            self.opt_res.primal_dual_plot_lists
        )

        # plot primal and dual objs
        first_kwargs: dict[str, Any] = dict(label=r"primal obj - $f(x^{(k)})$")
        first_kwargs.update(kwargs)
        [
            (
                None
                if np.all(np.isnan(primal_obj_list_list[member_idx]))
                else obj_val_axis.plot(
                    iter_list,
                    primal_obj_list_list[member_idx],
                    *args,
                    **(first_kwargs if member_idx == 0 else kwargs),
                )
            )
            for member_idx, iter_list in enumerate(iter_list_list)
        ]

        first_kwargs = dict(label=r"dual obj - $g(\lambda^{(k)}, \nu^{(k)})$")
        first_kwargs.update(kwargs)
        [
            (
                None
                if np.all(np.isnan(dual_objs_list_list[member_idx]))
                else obj_val_axis.plot(
                    iter_list,
                    dual_objs_list_list[member_idx],
                    *args,
                    **(first_kwargs if member_idx == 0 else kwargs),
                )
            )
            for member_idx, iter_list in enumerate(iter_list_list)
        ]

        if primal_gap_axis is not None:
            try:
                [
                    primal_gap_axis.semilogy(
                        iter_list,
                        primal_obj_list_list[member_idx] - self.opt_res.opt_prob.optimum_value,
                        *args,
                        **kwargs,
                    )
                    for member_idx, iter_list in enumerate(iter_list_list)
                ]
                primal_gap_axis.set_title(r"primal gap - $f(x^{(k)}) - p^\ast$", fontsize=10.0)
            except ValueUnknownException:
                pass

        if dual_gap_axis is not None:
            try:
                [
                    dual_gap_axis.semilogy(
                        iter_list,
                        self.opt_res.opt_prob.optimum_value - dual_objs_list_list[member_idx],
                        *args,
                        **kwargs,
                    )
                    for member_idx, iter_list in enumerate(iter_list_list)
                ]
                dual_gap_axis.set_title(
                    r"dual gap - $p^\ast- g(\lambda^{(k)}, \nu^{(k)})$", fontsize=10.0
                )
            except ValueUnknownException:
                pass

        for ax in [obj_val_axis, primal_gap_axis, dual_gap_axis]:
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
