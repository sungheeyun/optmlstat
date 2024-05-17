"""
base class for iterative optimization algorithms
"""

from abc import abstractmethod
from typing import Any

import numpy as np

from optmlstat.opt.opt_parameter import OptParams
from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.optalgs.optalg_base import OptAlgBase


class IterativeOptAlgBase(OptAlgBase):

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
