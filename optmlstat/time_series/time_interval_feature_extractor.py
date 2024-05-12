from datetime import datetime
from typing import Callable, List

from numpy import ndarray, logical_and, hstack
from pandas import DataFrame, Index

from optmlstat.time_series.feature_extractor_base import FeatureExtractorBase
from optmlstat.time_series.time_series_collection import TimeSeriesCollection


class TimeIntervalFeatureExtractor(FeatureExtractorBase):
    """
    Extracts statistics for a specific time interval for each time series
    """

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        feature_extractor: Callable[[ndarray], ndarray],
    ) -> None:
        self.start_time: datetime = start_time
        self.end_time: datetime = end_time
        self.feature_extractor: Callable[[ndarray], ndarray] = feature_extractor

    def get_features_from_time_series_collection(
        self, time_series_collection: TimeSeriesCollection
    ) -> ndarray:
        array_list: List[ndarray] = list()
        for time_series in time_series_collection.time_series_list:
            time_series_data_frame: DataFrame = time_series.time_series_data_frame
            date_time_index: Index = time_series_data_frame.index
            idx_array: ndarray = logical_and(
                date_time_index >= self.start_time, date_time_index < self.end_time
            )
            array_list.append(
                self.feature_extractor(time_series_data_frame.iloc[idx_array].values.ravel())
            )

        return hstack(array_list)
