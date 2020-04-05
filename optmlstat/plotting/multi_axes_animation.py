from typing import List, Dict, Any, Tuple
from functools import reduce

from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.animation as animation

from optmlstat.utils.interval import Interval


class MultiAxesAnimation(animation.TimedAnimation):
    """
    Performs simultaneous animation for multiple Axes.

    """

    def __init__(
        self,
        figure: Figure,
        axis_list: List[Axes],
        time_array_1d: ndarray,
        x_array_2d: ndarray,
        y_array_2d: ndarray,
        head_time_period: float = 1.0,
        **kwargs
    ) -> None:
        assert time_array_1d.ndim == 1
        assert x_array_2d.ndim == 2
        assert y_array_2d.ndim == 2

        assert len(axis_list) == x_array_2d.shape[1]
        assert len(axis_list) == y_array_2d.shape[1]

        assert time_array_1d.size == x_array_2d.shape[0]
        assert time_array_1d.size == y_array_2d.shape[0]

        assert head_time_period > 0.0

        self.time_array_1d: ndarray = time_array_1d.copy()
        self.x_array_2d: ndarray = x_array_2d.copy()
        self.y_array_2d: ndarray = y_array_2d.copy()
        self.head_time_period: float = head_time_period

        # TODO (2) control color, line width, etc. using constructor arguments.

        self.figure: Figure = figure
        self.axis_list: List[Axes] = axis_list
        self.name_line2d_dict_list: List[Dict[str, Line2D]] = [
            dict(
                line1=Line2D([], [], color="black"),
                line1a=Line2D([], [], color="red", linewidth=2),
                line1e=Line2D([], [], color="red", marker="o", markeredgecolor="r"),
            )
            for _ in self.axis_list
        ]

        axis_interval_dict: Dict[Axes, Tuple[Interval, Interval]] = dict()

        for axis_idx, axis in enumerate(self.axis_list):
            for line2d in self.name_line2d_dict_list[axis_idx].values():
                axis.add_line(line2d)

            x_array_1d: ndarray = self.x_array_2d[:, axis_idx]
            y_array_1d: ndarray = self.y_array_2d[:, axis_idx]

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

        self._drawn_artists: List[Line2D] = reduce(
            list.__add__, [list(name_line2d_dict.values()) for name_line2d_dict in self.name_line2d_dict_list]
        )

        _kwargs: Dict[str, Any] = dict(interval=10.0, blit=True)
        _kwargs.update(kwargs)
        animation.TimedAnimation.__init__(self, figure, **_kwargs)

    def _draw_frame(self, framedata):
        current_idx = framedata
        head = current_idx - 1
        head_slice = (self.time_array_1d > self.time_array_1d[current_idx] - self.head_time_period) & (
            self.time_array_1d < self.time_array_1d[current_idx]
        )

        for axis_idx, name_line2d_dict in enumerate(self.name_line2d_dict_list):
            x_array_1d: ndarray = self.x_array_2d[:, axis_idx]
            y_array_1d: ndarray = self.y_array_2d[:, axis_idx]
            name_line2d_dict["line1"].set_data(x_array_1d[:current_idx], y_array_1d[:current_idx])
            name_line2d_dict["line1a"].set_data(x_array_1d[head_slice], y_array_1d[head_slice])
            name_line2d_dict["line1e"].set_data(x_array_1d[head], y_array_1d[head])

    def new_frame_seq(self):
        return iter(range(self.time_array_1d.size))

    def _init_draw(self):
        for line2d in self._drawn_artists:
            line2d.set_data([], [])
