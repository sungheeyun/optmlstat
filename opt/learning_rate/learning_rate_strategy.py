from typing import Optional
from dataclasses import dataclass


@dataclass()
class LearningRateStrategy:
    """
    Learning rate strategy. This class is a default one; a constant learning rate.
    """
    constant_learning_rate: float

    def get_learning_rate(self, outer_iteration: Optional[int] = None, inner_iteration: Optional[int] = None) -> float:
        return self.constant_learning_rate
