"""
line search base
"""

from abc import abstractmethod
from typing import Callable

import numpy as np

from optmlstat.basic_modules.class_base import OMSClassBase


class LineSearchBase(OMSClassBase):
    @abstractmethod
    def search(
        self,
        fcn: Callable,
        x_array_2d: np.ndarray,
        search_dir_2d: np.ndarray,
        directional_deriv: np.ndarray,
    ) -> np.ndarray:
        pass
