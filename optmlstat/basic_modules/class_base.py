"""
formatter
"""

from abc import ABC
import json

from optmlstat.formatting import convert_data_for_json


class OMSClassBase(ABC):
    """
    The base class for all the classes in `optmlstat` package.

    """

    def to_json_data(self) -> int | float | str | dict | list:
        """
        Returns json data representing the class.
        This is mainly for data serialization.

        Returns
        -------
        json_data:
         json data object
        """
        return dict(class_category=self.__class__.__name__)

    def __repr__(self) -> str:
        # return json.dumps(self.to_json_data(), indent=2, default=convert_data_for_json)
        return json.dumps(self.to_json_data(), default=convert_data_for_json)
