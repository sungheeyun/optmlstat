from typing import Union
from abc import ABC, abstractmethod


class OptMLStatClassBase(ABC):
    """
    The base class for all the classes in `optmlstat` package.

    """
    @abstractmethod
    def to_json_data(self) -> Union[int, float, str, dict, list]:
        pass
