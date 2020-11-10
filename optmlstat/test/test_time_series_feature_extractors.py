from typing import List
import unittest

from numpy import array, ndarray, count_nonzero
from numpy.random import randn
from pandas import date_range, Timestamp, DataFrame, DatetimeIndex
from matplotlib.pyplot import subplots
from matplotlib.figure import Figure

from optmlstat.time_series.time_series import TimeSeries
from optmlstat.time_series.time_series_collection import TimeSeriesCollection
from optmlstat.time_series.time_interval_feature_extractor import TimeIntervalFeatureExtractor


class TestTimeSeriesFeatureExtractors(unittest.TestCase):

    NUM_TIME_POINTS: int = 100
    START_TIME: int = Timestamp("2020-01-01 06:00:00")
    INTERVAL_START_TIME: Timestamp = Timestamp("2020-01-01 06:10:00")
    INTERVAL_END_TIME: Timestamp = Timestamp("2020-01-01 06:20:00")
    FEATURES: ndarray = array([2, 3, 1], float)
    NOISE_SIZE: float = 0.1

    time_series_collection: TimeSeriesCollection

    @classmethod
    def setUpClass(cls) -> None:
        date_time_index: DatetimeIndex = date_range(cls.START_TIME, periods=cls.NUM_TIME_POINTS, freq="min")
        interval_idx_array: ndarray = (date_time_index >= cls.INTERVAL_START_TIME) & (
            date_time_index < cls.INTERVAL_END_TIME
        )

        time_series_list: List[TimeSeries] = list()
        for feature in cls.FEATURES:
            value_array: ndarray = randn(cls.NUM_TIME_POINTS)
            value_array[interval_idx_array] = cls.NOISE_SIZE * randn(count_nonzero(interval_idx_array)) + feature
            time_series: TimeSeries = TimeSeries(DataFrame(data=value_array, index=date_time_index))
            time_series_list.append(time_series)

        cls.time_series_collection = TimeSeriesCollection(time_series_list)

    def test_basic_feature_extractor(self) -> None:

        time_interval_feature_extractor: TimeIntervalFeatureExtractor = TimeIntervalFeatureExtractor(
            TestTimeSeriesFeatureExtractors.INTERVAL_START_TIME,
            TestTimeSeriesFeatureExtractors.INTERVAL_END_TIME,
            lambda array_1d: array([array_1d.mean()]),
        )

        figure: Figure
        axes_array: ndarray
        figure, axes_array = subplots(TestTimeSeriesFeatureExtractors.FEATURES.size, 1)

        for idx, time_series in enumerate(TestTimeSeriesFeatureExtractors.time_series_collection.time_series_list):
            time_series.time_series_data_frame.plot(ax=axes_array[idx])

        figure.show()

        features: ndarray = time_interval_feature_extractor.get_features_from_time_series_collection(
            TestTimeSeriesFeatureExtractors.time_series_collection
        )

        self.assertLess(abs(features - TestTimeSeriesFeatureExtractors.FEATURES).max(), 0.1)


if __name__ == "__main__":
    unittest.main()
