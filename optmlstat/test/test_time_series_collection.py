import sys
import unittest

from pandas import DataFrame, read_csv, Timestamp

from optmlstat.time_series.time_series import TimeSeries
from optmlstat.time_series.time_series_collection import TimeSeriesCollection

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

TIME_SERIES_DATA_TEXT: StringIO = StringIO(
    """time;value
    2020-1-1 00:01:00;10
    2020-1-1 00:02:00;20
    2020-1-1 00:03:00;30
    2020-1-1 00:06:00;40
    """
)


class TestTimeSeriesCollection(unittest.TestCase):
    def test_time_series_collection(self):
        data_frame: DataFrame = read_csv(TIME_SERIES_DATA_TEXT, sep=";", index_col=0)
        data_frame.index = data_frame.index.map(Timestamp)

        time_series: TimeSeries = TimeSeries(data_frame)

        time_series_collection: TimeSeriesCollection = TimeSeriesCollection([time_series] * 5)

        for time_series in time_series_collection.time_series_list:
            self.assertNotEquals(time_series.name, "time_series_0")


if __name__ == "__main__":
    unittest.main()
