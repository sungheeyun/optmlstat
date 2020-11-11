from datetime import datetime
from numpy.random import randn

from pandas import DataFrame

from optmlstat.time_series.time_series import TimeSeries
from optmlstat.time_series.time_series_generator.time_series_generator import TimeSeriesGenerator


class RandomTimeSeriesGenerator(TimeSeriesGenerator):
    """
    Base class for random time series generator classes. The default time series is generated from the standard
     Gaussian, i.e., normal distribution with zero mean and unit variance.
    """
    def __init__(self, start_time: datetime, num_time_points: int, unit_time: str) -> None:
        super(RandomTimeSeriesGenerator, self).__init__(start_time, num_time_points, unit_time)

    def generate_time_series(self) -> TimeSeries:
        return TimeSeries(DataFrame(randn(self.num_time_points), index=self.date_time_index))
