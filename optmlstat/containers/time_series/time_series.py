from pandas import DataFrame, Timestamp

from optmlstat.basic_modules.class_base import OptMLStatClassBase


class TimeSeries(OptMLStatClassBase):
    """
    Time Series Data.
    """

    def __init__(self, time_series_data_frame: DataFrame, start_time: Timestamp = None) -> None:
        """
        The constructor assumes that the time_series_data_fram is sorted by index (time) in increasing order.

        Parameters
        ----------
        time_series_data_frame: pandas.DataFrame
         pandas.DataFrame the index of which is pandas.TimeStamp
        """
        self.time_series_data_frame: DataFrame = time_series_data_frame.copy()
        self.start_time: Timestamp
        if start_time is None:
            self.start_time = time_series_data_frame.index[0]
        else:
            self.start_time = start_time
