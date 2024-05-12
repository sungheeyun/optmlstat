from typing import Union
from abc import ABC


class OMSClassBase(ABC):
    """
    The base class for all the classes in `optmlstat` package.

    """

    def to_json_data(self) -> Union[int, float, str, dict, list]:
        """
        Returns json data representing the class.
        This is mainly for data serialization.

        Returns
        -------
        json_data:
         json data object
        """
        return dict(class_category=self.__class__.__name__)
