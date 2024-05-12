import unittest

from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from numpy import ndarray
from pandas import Timestamp

from optmlstat.time_series.plotters.time_series_plotter import (
    TimeSeriesPlotter,
)
from optmlstat.time_series.time_series_generator.random_interval_time_series_generator import (
    RandomIntervalTimeSeriesGenerator,
)
from optmlstat.time_series.time_series_generator.random_time_series_generator import (
    RandomTimeSeriesGenerator,
)
from optmlstat.time_series.time_series_generator.time_series_generator import (
    TimeSeriesGenerator,
)


class TestTimeSeriesGenerator(unittest.TestCase):
    START_TIME: Timestamp = Timestamp("2020-01-01 06:00:00")
    NUM_TIME_POINTS: int = 100
    UNIT_TIME: str = "min"

    INTERVAL_START_TIME: Timestamp = Timestamp("2020-01-01 06:20:00")
    INTERVAL_END_TIME: Timestamp = Timestamp("2020-01-01 06:40:00")

    def test_simple_time_series_generators(self):
        time_series_generator: TimeSeriesGenerator = TimeSeriesGenerator(
            TestTimeSeriesGenerator.START_TIME,
            TestTimeSeriesGenerator.NUM_TIME_POINTS,
            TestTimeSeriesGenerator.UNIT_TIME,
        )

        random_time_series_generator: RandomTimeSeriesGenerator = RandomTimeSeriesGenerator(
            TestTimeSeriesGenerator.START_TIME,
            TestTimeSeriesGenerator.NUM_TIME_POINTS,
            TestTimeSeriesGenerator.UNIT_TIME,
        )

        random_interval_time_series_generator: RandomIntervalTimeSeriesGenerator = (
            RandomIntervalTimeSeriesGenerator(
                TestTimeSeriesGenerator.START_TIME,
                TestTimeSeriesGenerator.NUM_TIME_POINTS,
                TestTimeSeriesGenerator.UNIT_TIME,
                TestTimeSeriesGenerator.INTERVAL_START_TIME,
                TestTimeSeriesGenerator.INTERVAL_END_TIME,
                20.0,
                0.1,
            )
        )

        time_series_plotter: TimeSeriesPlotter = TimeSeriesPlotter()

        figure: Figure
        ax_array: ndarray

        figure, ax_array = subplots(3, 1)

        time_series_plotter.plot(time_series_generator.generate_time_series(), ax=ax_array[0])
        time_series_plotter.plot(
            random_time_series_generator.generate_time_series(), ax=ax_array[1]
        )
        time_series_plotter.plot(
            random_interval_time_series_generator.generate_time_series(),
            ax=ax_array[2],
        )

        figure.show()

        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
