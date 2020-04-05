from dataclasses import dataclass

from optmlstat.opt.iteration import Iteration


@dataclass()
class LearningRateStrategy:
    """
    Learning rate strategy. This class is a default one; a constant learning rate.
    """

    constant_learning_rate: float

    def get_learning_rate(self, iteration: Iteration) -> float:
        return self.constant_learning_rate
