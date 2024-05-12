"""
line search base
"""

from abc import abstractmethod

import numpy as np

from optmlstat.basic_modules.class_base import OMSClassBase
from optmlstat.functions.function_base import FunctionBase


class LineSearchBase(OMSClassBase):
    @abstractmethod
    def search(
        self, fcn: FunctionBase, x_array_2d: np.ndarray, search_dir_2d: np.ndarray
    ) -> np.ndarray:
        pass
