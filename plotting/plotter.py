from typing import List, Union, Iterable

from numpy import ndarray
from matplotlib.axes import Axes


def plot_1d_data(axes_list: Union[List[Axes], Axes], x_array_2d: ndarray, y_array_2d: ndarray, *args, **kwargs) -> None:
    """
    Plot data for 1-dimensional X and multi-dimensional Y.
    It plots each y for each Axis, hence the number of outputs should be no less than the number of Axes.

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
