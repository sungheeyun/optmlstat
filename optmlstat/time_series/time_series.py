from datetime import datetime
from typing import Set, Optional

from pandas import DataFrame

from optmlstat.basic_modules.class_base import OMSClassBase


class TimeSeries(OMSClassBase):
    """
    Time Series Data.
    """

    time_series_number: int = 0
    time_series_name_set: Set[str] = set()

    @classmethod
    def add_time_series_name(cls, time_series_name: str) -> None:
        cls.time_series_name_set.add(time_series_name)

    @classmethod
    def get_new_time_series_name(cls) -> str:
        cnt: int = 0
        while True:
            time_series_name: str = f"time_series_{cls.time_series_number}"
            cls.time_series_number += 1
            if time_series_name not in cls.time_series_name_set:
                break

            if cnt > 100000:
                raise Exception("Infinite loop error!")

        return time_series_name

    def __init__(
        self,
        time_series_data_frame: DataFrame,
        start_time: Optional[datetime] = None,
        name: str | None = None,
    ) -> None:
        """
        The constructor assumes that the time_series_data_fram is sorted
        by index (time) in increasing order.

        Parameters
        ----------
        time_series_data_frame: pandas.DataFrame
         pandas.DataFrame the (contents of) indices of which are of datetime.datetime type.
        """
        self.time_series_data_frame: DataFrame = time_series_data_frame.copy()
        self.start_time: datetime
        if start_time is None:
            self.start_time = time_series_data_frame.index[0]
        else:
            self.start_time = start_time

        self.name: str
        if name is None:
            self.name = self.get_new_time_series_name()

        self.add_time_series_name(self.name)
