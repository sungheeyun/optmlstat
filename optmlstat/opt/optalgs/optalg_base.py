"""
base class for optimization algorithms
"""

from abc import abstractmethod
from typing import Any

import numpy as np

from optmlstat.basic_modules.class_base import OMSClassBase
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_parameter import OptParams


class OptAlgBase(OMSClassBase):
    """
    Optimization Algorithm
    """

    @abstractmethod
    def solve(
        self,
        opt_prob: OptProb,
        opt_param: OptParams,
        verbose: bool,
        /,
        *,
        initial_x_array_2d: np.ndarray,
        initial_lambda_array_2d: np.ndarray | None = None,
        initial_nu_array_2d: np.ndarray | None = None,
    ) -> Any:
        pass
