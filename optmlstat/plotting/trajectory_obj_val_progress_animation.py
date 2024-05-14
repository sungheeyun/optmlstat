"""
animation showing optimization progress
"""

from functools import reduce
from typing import Any

import matplotlib.animation as animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from optmlstat.utils.interval import Interval


class TrajectoryObjValProgressAnimation(animation.TimedAnimation):
    """
    Performs simultaneous animation for multiple Axes.

    """

    def __init__(
        self,
        figure: Figure,
        axis_list: list[Axes],
        iter_axes: list[Axes],
        /,
        time_array_1d: np.ndarray,
        x_array_2d: np.ndarray,
        y_array_2d: np.ndarray,
        head_time_period: float = 3.0,
        **kwargs,
    ) -> None:
        assert time_array_1d.ndim == 1
        assert x_array_2d.ndim == 2
        assert y_array_2d.ndim == 2

        assert len(axis_list) == x_array_2d.shape[1]
        assert len(axis_list) == y_array_2d.shape[1]

        assert time_array_1d.size == x_array_2d.shape[0]
        assert time_array_1d.size == y_array_2d.shape[0]

        assert head_time_period > 0.0

        self.time_array_1d: np.ndarray = time_array_1d.copy()
        self.x_array_2d: np.ndarray = x_array_2d.copy()
        self.y_array_2d: np.ndarray = y_array_2d.copy()
        self.head_time_period: float = head_time_period

        # TODO (2) control color, line width, etc. using constructor arguments.

        # for ax in axis_list:
        #     ax.axis("equal")
        self.name_line2d_dict_list: list[dict[str, Line2D]] = [
            dict(
                line1=Line2D([], [], color="black", linewidth=1, alpha=0.2),
                line1a=Line2D([], [], color="red", linewidth=1, alpha=0.5),
                line1e=Line2D([], [], color="red", marker="o", markeredgecolor="r", markersize=4),
            )
            for _ in axis_list
        ]

        self.iter_axes_ylims: list[tuple[float, float]] = [ax.get_ylim() for ax in iter_axes]
        self.ver_line_list_list: list[list[Line2D]] = [
            [Line2D([], [], color="black", linewidth=0.5, alpha=0.5) for _ in self.time_array_1d]
            for _ in iter_axes
        ]
        for idx, iter_axis in enumerate(iter_axes):
            ver_line_list: list[Line2D] = self.ver_line_list_list[idx]
            for ver_line in ver_line_list:
                iter_axis.add_line(ver_line)

        axis_interval_dict: dict[Axes, tuple[Interval, Interval]] = dict()

        for axis_idx, axis in enumerate(axis_list):
            for line2d in self.name_line2d_dict_list[axis_idx].values():
                axis.add_line(line2d)

            x_array_1d: np.ndarray = self.x_array_2d[:, axis_idx]
            y_array_1d: np.ndarray = self.y_array_2d[:, axis_idx]

            xlim: Interval = Interval(x_array_1d.min(), x_array_1d.max())
            ylim: Interval = Interval(y_array_1d.min(), y_array_1d.max())

            if axis in axis_interval_dict:
                axis_interval_dict[axis][0].update(xlim)
                axis_interval_dict[axis][1].update(ylim)
            else:
                axis_interval_dict[axis] = (xlim, ylim)

        for axis, (xlim, ylim) in axis_interval_dict.items():
            axis.set_xlim(xlim.lower_bound, xlim.upper_bound)
            axis.set_ylim(ylim.lower_bound, ylim.upper_bound)

        self._drawn_artists: list[Line2D] = reduce(
            list.__add__,
            [list(name_line2d_dict.values()) for name_line2d_dict in self.name_line2d_dict_list],
        ) + reduce(list.__add__, self.ver_line_list_list)

        _kwargs: dict[str, Any] = dict(interval=100.0, blit=True)
        _kwargs.update(kwargs)
        animation.TimedAnimation.__init__(self, figure, **_kwargs)

    def _draw_frame(self, frame_data) -> None:
        current_idx = frame_data
        head = current_idx
        head_slice = (
            self.time_array_1d > self.time_array_1d[current_idx] - self.head_time_period
        ) & (self.time_array_1d <= self.time_array_1d[current_idx])

        for axis_idx, name_line2d_dict in enumerate(self.name_line2d_dict_list):
            x_array_1d: np.ndarray = self.x_array_2d[:, axis_idx]
            y_array_1d: np.ndarray = self.y_array_2d[:, axis_idx]
            name_line2d_dict["line1"].set_data(
                x_array_1d[: current_idx + 1], y_array_1d[: current_idx + 1]
            )
            name_line2d_dict["line1a"].set_data(x_array_1d[head_slice], y_array_1d[head_slice])
            name_line2d_dict["line1e"].set_data(x_array_1d[head], y_array_1d[head])

        for axis_idx, ver_line_list in enumerate(self.ver_line_list_list):
            ylim: tuple[float, float] = self.iter_axes_ylims[axis_idx]
            for iter_idx, ver_line in enumerate(ver_line_list):
                if iter_idx == head:
                    ver_line.set_data(float(iter_idx) * np.ones(2), ylim)
                else:
                    ver_line.set_data([], [])

    def new_frame_seq(self) -> Any:
        return iter(range(self.time_array_1d.size))

    def _init_draw(self) -> None:
        for line2d in self._drawn_artists:
            line2d.set_data([], [])
