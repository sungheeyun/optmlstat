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
    **kwargs,
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


NUM_PNTS_PER_AXIS_FOR_CONTOUR: int = 100


def plot_fcn_contour(
    ax: Axes,
    fcn: FunctionBase,
    project_array: np.ndarray,
    /,
    *,
    fcn_num: int = 0,
    center: np.ndarray | None = None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    **kwargs,
) -> None:
    assert fcn.num_inputs > 1, fcn.num_inputs
    assert fcn.num_inputs == project_array.shape[0], (fcn.num_inputs, project_array.shape)
    assert 0 <= fcn_num < fcn.num_outputs, (fcn_num, fcn.num_outputs)

    _center: np.ndarray = np.zeros(fcn.num_inputs) if center is None else center

    _center_orth_coor: np.ndarray = np.dot(_center, project_array)  # (n,) x (n,2) = (2,)

    x: np.ndarray = np.linspace(xlim[0], xlim[1], NUM_PNTS_PER_AXIS_FOR_CONTOUR)
    y: np.ndarray = np.linspace(ylim[0], ylim[1], NUM_PNTS_PER_AXIS_FOR_CONTOUR)
    X, Y = np.meshgrid(x, y)

    Z = fcn.eval(
        np.dot(np.array([X, Y]).reshape((2, -1)).T - _center_orth_coor, project_array.T) + _center
    ).reshape(X.shape)

    ax.contour(X, Y, Z, **kwargs)
    # ax.set_ylim(ylim)
