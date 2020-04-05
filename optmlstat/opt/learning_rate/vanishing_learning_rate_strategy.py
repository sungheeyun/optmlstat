from numpy import power

from optmlstat.opt.iteration import Iteration
from optmlstat.opt.learning_rate.learning_rate_strategy import LearningRateStrategy


class VanishingLearningRateStrategy(LearningRateStrategy):
    """
    Implements vanishing (or decreasing) learning rate strategy (depending on outer iteration).
    The learning rate is

      eta = a / (outer_iteration + b)^k

    where k is a positive exponent controlling decreasing rate.
    """

    def __init__(self, initial_value: float, exponent: float, half_life: int) -> None:
        """
        The learning rate is given by

         eta = a / (outer_iteration + b)^k

        where k is the exponent. The other two constants a and b are determined by initial_value and half_life,
        i.e.,

        a / (1 + b)^k = initial_value
        a / (half_life + b)^k = initial_value / 2

        Therefore

        (half_life + b)/(1 + b) = 2^(1/k) <=> (2^(1/k) - 1) b = half_life - 2^(1/k)
         <=> b = (half_life - 2^(1/k)) / (2^(1/k) - 1)

        and

        a = initial_value * (1 + b)^k

        Parameters
        ----------
        initial_value:
         initial_value
        exponent:
         exponent
        half_life:
         half_life
        """
        assert exponent > 0.0
        assert half_life > 1

        # In this class, self.constant_learning_rate is the initial value.
        super(VanishingLearningRateStrategy, self).__init__(initial_value)

        self.exponent: float = exponent
        self.half_life: int = half_life

        tmp_val: float = power(2.0, 1.0 / exponent)

        self.constant_b: float = float(self.half_life - tmp_val) / (tmp_val - 1.0)
        self.constant_a: float = float(self.constant_learning_rate) * power(1.0 + self.constant_b, self.exponent)

    def get_learning_rate(self, iteration: Iteration) -> float:
        assert iteration.outer_iteration > 0
        return self.constant_a / power(float(iteration.outer_iteration) + self.constant_b, self.exponent)
