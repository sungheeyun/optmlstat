"""
plotting functions
"""

from typing import List, Union, Iterable

import numpy as np
from matplotlib.axes import Axes

from optmlstat.functions.function_base import FunctionBase


def plot_1d_data(
    axes_list: Union[List[Axes], Axes],
    x_array_2d: np.ndarray,
    y_array_2d: np.ndarray,
    *args,
    **kwargs
) -> None:
    """
    Plot data for 1-dimensional X and multi-dimensional Y.
    It plots each y for each Axis, hence the number of outputs
    should be no less than the number of Axes.

    :param axes_list:
     List of Axes
    :param x_array_2d:
     N-by-n array for X. (Only the first column is used.)
    :param y_array_2d:
     N-by-m array for Y.
    """
    if not isinstance(axes_list, Iterable):
        axes_list = [axes_list]

    for idx, axis in enumerate(axes_list):
        axis.plot(x_array_2d[:, 0], y_array_2d[:, idx], *args, **kwargs)


def plot_fcn_contour(ax: Axes, fcn: FunctionBase, fcn_num: int = 0, **kwargs) -> None:
    assert fcn.num_inputs == 2, fcn.num_inputs
    assert 0 <= fcn_num < fcn.num_outputs, (fcn_num, fcn.num_outputs)

    xlim: tuple[float, float] = kwargs.pop("xlim", (-1.0, 1.0))
    ylim: tuple[float, float] = kwargs.pop("ylim", (-1.0, 1.0))
    num_pnts_per_axis: int = kwargs.pop("num_pnts_per_axis", 100)

    x: np.ndarray = np.linspace(xlim[0], xlim[1], num_pnts_per_axis)
    y: np.ndarray = np.linspace(ylim[0], ylim[1], num_pnts_per_axis)
    X, Y = np.meshgrid(x, y)

    Z = fcn.eval(np.array([X, Y]).reshape((2, -1)).T).reshape(X.shape)

    ax.contour(X, Y, Z, **kwargs)
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
