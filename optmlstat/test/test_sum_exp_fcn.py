"""
test sum_exp functions
"""

import unittest

from freq_used.plotting import get_figure
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from optmlstat.functions.function_base import FunctionBase
from optmlstat.functions.basic_functions.log_sum_exp import LogSumExp
from optmlstat.plotting.plotter import plot_fcn_contour


class TestSumExpFcn(unittest.TestCase):
    def test_sum_exp_fcn(self):
        sum_exp: LogSumExp = LogSumExp(
            [[[1.0, 3.0], [1.0, -3.0], [-1.0, 0.0]]], -0.1 * np.ones((1, 3))
        )

        fig: Figure = get_figure(1, 1, axis_width=4.0, axis_height=4.0)

        ax: Axes = fig.get_axes()[0]
        # xlim = -10.0, 10.0
        # ylim = -10.0, 10.0
        xlim = -3.0, 3.0
        ylim = -3.0, 3.0
        plot_fcn_contour(ax, sum_exp, xlim=xlim, ylim=ylim, levels=10)
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

        # plt.show()
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
