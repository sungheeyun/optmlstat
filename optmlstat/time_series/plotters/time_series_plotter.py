from matplotlib.axes import Axes

from optmlstat.basic_modules.class_base import OptMLStatClassBase
from optmlstat.time_series.time_series import TimeSeries


class TimeSeriesPlotter(OptMLStatClassBase):
    """
    Plots TimeSeries.
    """

    def plot(self, time_series: TimeSeries, ax: Axes, **kwargs) -> None:
        time_series.time_series_data_frame.plot(ax=ax, **kwargs)

        for x_major_tick_label in ax.xaxis.get_majorticklabels():
            x_major_tick_label.set_rotation(45.0)

        for x_minor_tick_label in ax.xaxis.get_minorticklabels():
            x_minor_tick_label.set_rotation(45.0)
