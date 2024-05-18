"""
optimization parameters
"""

from dataclasses import dataclass

from optmlstat.opt.learning_rate.learning_rate_strategy import LearningRateStrategy


@dataclass(frozen=True)
class OptParams:
    _learning_rate_strategy: float | LearningRateStrategy = 0.01
    max_num_outer_iterations: int = 100
    max_num_inner_iterations: int | None = None
    back_tracking_line_search_alpha: float = 0.25
    back_tracking_line_search_beta: float = 0.5
    abs_tolerance_on_optimality: float = 1e-8
    rel_tolerance_on_optimality: float = 1e-5
    tolerance_on_grad: float = 1e-6
    tolerance_on_newton_dec: float = 1e-9

    @property
    def learning_rate_strategy(self) -> LearningRateStrategy:
        return (
            self._learning_rate_strategy
            if isinstance(self._learning_rate_strategy, LearningRateStrategy)
            else LearningRateStrategy(self._learning_rate_strategy)
        )
