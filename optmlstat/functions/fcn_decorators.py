"""
decorators for optimization
mostly for checking requirements for each solver
"""

from functools import wraps
from typing import Callable

import numpy as np

from optmlstat.functions.function_base import FunctionBase


def fcn_evaluator(func: Callable) -> Callable:
    @wraps(func)
    def fcn_evaluator_wrapper(self: FunctionBase, x_array_2d: np.ndarray) -> np.ndarray:
        assert x_array_2d.shape[1] == self.num_inputs, (
            x_array_2d.shape,
            self.num_inputs,
        )
        return func(self, x_array_2d)

    return fcn_evaluator_wrapper
