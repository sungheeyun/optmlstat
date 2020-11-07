import sys
import unittest

from pandas import read_csv, DataFrame, Timestamp
from matplotlib.pyplot import subplots
from matplotlib.figure import Figure
from matplotlib.axes import Axes

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


class TestTimeSeries(unittest.TestCase):
    def test_basic_time_series(self):
        data_frame: DataFrame = read_csv(TIME_SERIES_DATA_TEXT, sep=";", index_col=0)

        data_frame.index = data_frame.index.map(Timestamp)

        fig: Figure
        ax: Axes

        fig, ax = subplots()
        data_frame.plot(ax=ax, marker="o", linestyle="-")

        for x_major_tick_label in ax.get_xmajorticklabels():
            x_major_tick_label.set_rotation(45)

        fig.show()

        self.assertTrue(isinstance(data_frame, DataFrame))


if __name__ == "__main__":
    unittest.main()
