"""
optimization parameters
"""

from dataclasses import dataclass

from optmlstat.opt.learning_rate.learning_rate_strategy import LearningRateStrategy


@dataclass(frozen=True)
class OptParams:
    _learning_rate_strategy: float | LearningRateStrategy
    max_num_outer_iterations: int
    max_num_inner_iterations: int | None = None
    back_tracking_line_search_alpha: float | None = None
    back_tracking_line_search_beta: float | None = None
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
