from typing import List

from optmlstat.basic_modules.class_base import OptMLStatClassBase
from optmlstat.time_series.time_series import TimeSeries


class TimeSeriesCollection(OptMLStatClassBase):
    """
    Ordered collection of TimeSeries instances.
    """

    def __init__(self, time_series_list: List[TimeSeries]) -> None:
        self.time_series_list: List[TimeSeries] = time_series_list.copy()
