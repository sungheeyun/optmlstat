import unittest
from datetime import datetime
from typing import List

from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from numpy import array, ndarray
from pandas import Timestamp

from optmlstat.time_series.plotters.time_series_plotter import (
    TimeSeriesPlotter,
)
from optmlstat.time_series.time_interval_feature_extractor import (
    TimeIntervalFeatureExtractor,
)
from optmlstat.time_series.time_series import TimeSeries
from optmlstat.time_series.time_series_collection import TimeSeriesCollection
from optmlstat.time_series.time_series_generator.random_interval_time_series_generator import (
    RandomIntervalTimeSeriesGenerator,
)


class TestTimeSeriesFeatureExtractors(unittest.TestCase):
    START_TIME: datetime = Timestamp("2020-01-01 06:00:00")
    NUM_TIME_POINTS: int = 100
    TIME_UNIT: str = "min"
    INTERVAL_START_TIME: Timestamp = Timestamp("2020-01-01 06:20:00")
    INTERVAL_END_TIME: Timestamp = Timestamp("2020-01-01 06:50:00")
    FEATURES: ndarray = array([2, 3, 1], float)
    NOISE_SIZE: float = 0.5

    time_series_collection: TimeSeriesCollection = TimeSeriesCollection([])

    @classmethod
    def setUpClass(cls) -> None:
        time_series_list: List[TimeSeries] = list()
        for feature in cls.FEATURES:
            time_series = RandomIntervalTimeSeriesGenerator(
                TestTimeSeriesFeatureExtractors.START_TIME,
                TestTimeSeriesFeatureExtractors.NUM_TIME_POINTS,
                TestTimeSeriesFeatureExtractors.TIME_UNIT,
                TestTimeSeriesFeatureExtractors.INTERVAL_START_TIME,
                TestTimeSeriesFeatureExtractors.INTERVAL_END_TIME,
                feature,
                cls.NOISE_SIZE,
            ).generate_time_series()
            time_series_list.append(time_series)

        cls.time_series_collection = TimeSeriesCollection(time_series_list)

    def test_basic_feature_extractor(self) -> None:

        time_interval_feature_extractor: TimeIntervalFeatureExtractor = (
            TimeIntervalFeatureExtractor(
                TestTimeSeriesFeatureExtractors.INTERVAL_START_TIME,
                TestTimeSeriesFeatureExtractors.INTERVAL_END_TIME,
                lambda array_1d: array([array_1d.mean()]),
            )
        )

        time_series_plotter: TimeSeriesPlotter = TimeSeriesPlotter()

        figure: Figure
        axes_array: ndarray
        figure, axes_array = subplots(TestTimeSeriesFeatureExtractors.FEATURES.size, 1)

        for idx, time_series in enumerate(
            TestTimeSeriesFeatureExtractors.time_series_collection.time_series_list
        ):
            time_series_plotter.plot(time_series, ax=axes_array[idx])

        figure.show()

        features: ndarray = (
            time_interval_feature_extractor.get_features_from_time_series_collection(
                TestTimeSeriesFeatureExtractors.time_series_collection
            )
        )

        print(TestTimeSeriesFeatureExtractors.FEATURES)
        print(features)

        self.assertLess(abs(features - TestTimeSeriesFeatureExtractors.FEATURES).max(), 1.0)


if __name__ == "__main__":
    unittest.main()
