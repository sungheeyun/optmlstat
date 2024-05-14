"""
base class for optimization algorithms
"""

from abc import abstractmethod
from typing import Any
import time

import numpy as np

from optmlstat.basic_modules.class_base import OMSClassBase
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_parameter import OptParams


class OptAlgBase(OMSClassBase):
    """
    Optimization Algorithm
    """

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
        t0: float = time.time()
        opt_res: Any = self._solve(
            opt_prob,
            opt_param,
            verbose,
            initial_x_array_2d=initial_x_array_2d,
            initial_lambda_array_2d=initial_lambda_array_2d,
            initial_nu_array_2d=initial_nu_array_2d,
        )
        opt_res.solve_time = time.time() - t0
        return opt_res

    @abstractmethod
    def _solve(
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
