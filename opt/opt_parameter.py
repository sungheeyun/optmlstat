from dataclasses import dataclass
from typing import Optional, Union

from opt.learning_rate.learning_rate_strategy import LearningRateStrategy


@dataclass(frozen=True)
class OptimizationParameter:
    _learning_rate_strategy: Union[float, LearningRateStrategy]
    max_num_outer_iterations: int
    max_num_inner_iterations: Optional[int] = None
    back_tracking_line_search_parameter_alpha: Optional[float] = None
    back_tracking_line_search_parameter_beta: Optional[float] = None
    abs_tolerance_on_optimality: float = 0.0
    rel_tolerance_on_optimality: float = 0.0
    tolerance_on_grad: float = 0.0

    @property
    def learning_rate_strategy(self) -> LearningRateStrategy:
        return (
            self._learning_rate_strategy
            if isinstance(self._learning_rate_strategy, LearningRateStrategy)
            else LearningRateStrategy(self._learning_rate_strategy)
        )
