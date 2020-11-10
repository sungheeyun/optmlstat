from abc import abstractmethod
from typing import List

from numpy import ndarray, concatenate

from optmlstat.basic_modules.class_base import OptMLStatClassBase
from optmlstat.time_series.time_series_collection import TimeSeriesCollection


class FeatureExtractorBase(OptMLStatClassBase):
    """
    Base class for classes for time series feature extractions.
    """

    @abstractmethod
    def get_features_from_time_series_collection(self, times_series_collection: TimeSeriesCollection) -> ndarray:
        """
        Parameters
        ----------
        times_series_collection: optmlstat.time_series.time_series_collecction.TimeSeriesCollection
         time series collection from which features are extracted.

        Returns
        -------
        feature_array: numpy.ndarray
         1-d numpy.ndarray containing features
        """
        pass

    def get_features_from_multiple_time_series_collections(
        self, time_series_collection_list: List[TimeSeriesCollection]
    ) -> ndarray:
        """
        Parameters
        ----------
        time_series_collection_list: list of optmlstat.time_series.time_series_collecction.TimeSeriesCollection
         list of time series collection from which list of features are extracted.

        Returns
        -------
        feature_array: numpy.ndarray
         2-d numpy.ndarray containing list of features
        """
        return concatenate(
            [
                self.get_features_from_time_series_collection(times_series)
                for times_series in time_series_collection_list
            ]
        )
