from datetime import datetime

from numpy import zeros
from pandas import date_range, DatetimeIndex, DataFrame

from optmlstat.basic_modules.class_base import OptMLStatClassBase
from optmlstat.time_series.time_series import TimeSeries


class TimeSeriesGenerator(OptMLStatClassBase):
    """
    Base class for time series generating classes.
    """

    def __init__(self, start_time: datetime, num_time_points: int, unit_time: str) -> None:
        self.start_time: datetime = start_time
        self.num_time_points: int = num_time_points
        self.time_unit: str = unit_time

        self.date_time_index: DatetimeIndex = date_range(
            self.start_time, periods=self.num_time_points, freq=self.time_unit
        )

    def generate_time_series(self) -> TimeSeries:
        return TimeSeries(DataFrame(zeros(self.num_time_points), index=self.date_time_index))
