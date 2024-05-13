"""
decorators for optimization
mostly for checking requirements for each solver
"""

from functools import wraps
from typing import Callable, Any

import numpy as np


def fcn_evaluator(func: Callable) -> Callable:
    @wraps(func)
    def fcn_evaluator_wrapper(self: Any, x_array_2d: np.ndarray) -> np.ndarray:
        assert self.num_inputs is None or x_array_2d.shape[1] == self.num_inputs, (
            x_array_2d.shape,
            self.num_inputs,
        )
        return func(self, x_array_2d)

    return fcn_evaluator_wrapper


def differentiable_fcn_evaluator(func: Callable) -> Callable:
    @wraps(func)
    def differentiable_fcn_evaluator_wrapper(self: Any, x_array_2d: np.ndarray) -> np.ndarray:
        assert self.is_differentiable, self.is_differentiable
        return func(self, x_array_2d)

    return differentiable_fcn_evaluator_wrapper


def twice_differentiable_fcn_evaluator(func: Callable) -> Callable:
    @wraps(func)
    def twice_differentiable_fcn_evaluator_wrapper(self: Any, x_array_2d: np.ndarray) -> np.ndarray:
        assert self.is_twice_differentiable, self.is_twice_differentiable
        return func(self, x_array_2d)

    return twice_differentiable_fcn_evaluator_wrapper
