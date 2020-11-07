from typing import Callable
import unittest

from numpy import ndarray, linspace, arctan
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import pyplot as plt

from optmlstat.functions.unit_functions import sigmoid


class TestUnitFunctions(unittest.TestCase):
    num_plotting_points: int = 100
    x_min: float = -5.0
    x_max: float = 5.0

    def test_sigmoid_function(self):
        TestUnitFunctions._draw_unit_function(sigmoid)
        self.assertTrue(True)

    def test_atan_function(self):
        TestUnitFunctions._draw_unit_function(arctan)
        self.assertTrue(True)

    @staticmethod
    def _draw_unit_function(ufcn: Callable) -> None:

        x_array_1d: ndarray = linspace(
            TestUnitFunctions.x_min, TestUnitFunctions.x_max, TestUnitFunctions.num_plotting_points
        )

        figure: Figure
        axes: Axes
        figure, axes = plt.subplots()
        axes.plot(x_array_1d, ufcn(x_array_1d))
        axes.set_title(ufcn)
        figure.show()


if __name__ == "__main__":
    unittest.main()
