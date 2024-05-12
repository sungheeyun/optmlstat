"""
"""

from datetime import datetime

from numpy import logical_and, ndarray, count_nonzero
from numpy.random import randn
from pandas import Index

from optmlstat.time_series.time_series import TimeSeries
from optmlstat.time_series.time_series_generator.random_time_series_generator import (
    RandomTimeSeriesGenerator,
)


class RandomIntervalTimeSeriesGenerator(RandomTimeSeriesGenerator):
    """
    Randomly generate times series following standard Gaussian,
    i.e., Gaussian with zero mean and unit variance
    with exception of one interval
    where the time series is generated with Gaussian with specified mean and variance.
    """

    def __init__(
        self,
        start_time: datetime,
        num_time_points: int,
        unit_time: str,
        interval_start_time: datetime,
        interval_end_time: datetime,
        average: float,
        std: float,
    ) -> None:
        super(RandomIntervalTimeSeriesGenerator, self).__init__(
            start_time, num_time_points, unit_time
        )

        self.interval_start_time: datetime = interval_start_time
        self.interval_end_time: datetime = interval_end_time
        self.average: float = average
        self.std: float = std

    def generate_time_series(self) -> TimeSeries:
        time_series: TimeSeries = super(
            RandomIntervalTimeSeriesGenerator, self
        ).generate_time_series()

        index: Index = time_series.time_series_data_frame.index
        interval_index: ndarray = logical_and(
            index >= self.interval_start_time, index < self.interval_end_time
        )

        time_series.time_series_data_frame.iloc[interval_index, 0] = (
            self.average + self.std * randn(count_nonzero(interval_index))
        )

        return time_series
