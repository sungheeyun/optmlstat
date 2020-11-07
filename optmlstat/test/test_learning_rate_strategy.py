from typing import List
import unittest
from logging import Logger, getLogger

from freq_used.logging_utils import set_logging_basic_config
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from optmlstat.opt.iteration import Iteration
from optmlstat.opt.learning_rate.learning_rate_strategy import LearningRateStrategy
from optmlstat.opt.learning_rate.vanishing_learning_rate_strategy import VanishingLearningRateStrategy

logger: Logger = getLogger()


class TestLearningRateStrategy(unittest.TestCase):
    max_outer_iteration: int = 100
    initial_value: float = 0.1
    half_life: int = 20
    exponent: float = 1.0

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_constant_learning_rate_strategy(self) -> None:
        constant_learning_rate_strategy: LearningRateStrategy = LearningRateStrategy(
            TestLearningRateStrategy.initial_value
        )
        for outer_iteration in range(TestLearningRateStrategy.max_outer_iteration + 1):
            iteration: Iteration = Iteration(outer_iteration)
            self.assertEqual(
                constant_learning_rate_strategy.get_learning_rate(iteration), TestLearningRateStrategy.initial_value
            )

    def test_vanishing_learning_rate_strategy(self) -> None:
        vanishing_learning_rate_strategy: VanishingLearningRateStrategy = VanishingLearningRateStrategy(
            TestLearningRateStrategy.initial_value,
            TestLearningRateStrategy.exponent,
            TestLearningRateStrategy.half_life,
        )

        figure: Figure
        axis: Axes
        figure, axis = plt.subplots()

        iteration_list: List[Iteration] = [
            Iteration(outer_iteration) for outer_iteration in range(1, TestLearningRateStrategy.max_outer_iteration + 1)
        ]
        axis.plot(
            [iteration.outer_iteration for iteration in iteration_list],
            [vanishing_learning_rate_strategy.get_learning_rate(iteration) for iteration in iteration_list],
            "-",
        )
        figure.show()

        self.assertEqual(
            vanishing_learning_rate_strategy.get_learning_rate(Iteration(1)), TestLearningRateStrategy.initial_value
        )
        self.assertEqual(
            vanishing_learning_rate_strategy.get_learning_rate(Iteration(TestLearningRateStrategy.half_life)),
            0.5 * TestLearningRateStrategy.initial_value,
        )


if __name__ == "__main__":
    unittest.main()
