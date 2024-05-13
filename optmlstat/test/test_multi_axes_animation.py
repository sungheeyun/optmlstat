import unittest
import os

import numpy as np
from numpy import ndarray, vstack
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import pyplot as plt

from optmlstat.plotting.multi_axes_animation import MultiAxesAnimation

OUTPUT_DIR: str = os.path.join(os.curdir, "output")


class TestMultiAxesAnimation(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        pass
        # plt.show()

    def test_multi_axes_animation(self) -> None:
        fig: Figure = plt.figure()
        ax1: Axes = fig.add_subplot(1, 2, 1)
        ax2: Axes = fig.add_subplot(2, 2, 2)
        ax3: Axes = fig.add_subplot(2, 2, 4)

        t_array_1d: ndarray = np.linspace(0, 80, 400)
        x_array_1d: ndarray = np.cos(2 * np.pi * t_array_1d / 10.0)
        y_array_1d: ndarray = np.sin(2 * np.pi * t_array_1d / 10.0)
        z_array_1d: ndarray = 10 * t_array_1d

        x_array_2d: ndarray = vstack([x_array_1d, y_array_1d, x_array_1d]).T
        y_array_2d: ndarray = vstack([y_array_1d, z_array_1d, z_array_1d]).T

        subplot_animation: MultiAxesAnimation = MultiAxesAnimation(
            fig,
            [ax1, ax2, ax3],
            [ax1],
            time_array_1d=t_array_1d,
            x_array_2d=x_array_2d,
            y_array_2d=y_array_2d,
            interval=5,
        )

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.set_xlabel("y")
        ax2.set_ylabel("z")
        ax3.set_xlabel("x")
        ax3.set_ylabel("z")

        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-2, 2)
        ax1.set_aspect("equal", "datalim")
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(0, 800)
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(0, 800)

        # subplot_animation.save(os.path.join(OUTPUT_DIR, "animation_test.mp4"))
        subplot_animation

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
