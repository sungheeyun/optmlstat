"""
plotting functions
"""

from typing import List, Union, Iterable

import numpy as np
from matplotlib.axes import Axes

from optmlstat.functions.function_base import FunctionBase
from optmlstat.functions.basic_functions.affine_function import AffineFunction


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


def relax_axis(axis: Axes, ratio: float = 0.1) -> None:
    xlim: tuple[float, float] = axis.get_xlim()
    ylim: tuple[float, float] = axis.get_ylim()

    axis.set_xlim(
        ((1.0 + ratio) * xlim[0] - ratio * xlim[1], (1.0 + ratio) * xlim[1] - ratio * xlim[0])
    )
    axis.set_ylim(
        ((1.0 + ratio) * ylim[0] - ratio * ylim[1], (1.0 + ratio) * ylim[1] - ratio * ylim[0])
    )


def plot_fcn_contour(
    ax: Axes,
    fcn: FunctionBase,
    project_2d: np.ndarray,
    /,
    *,
    fcn_num: int = 0,
    center: np.ndarray | None = None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    eq_cnst_fcn: FunctionBase | None = None,
    ineq_cnst_fcn: FunctionBase | None = None,
    **kwargs,
) -> None:
    assert fcn.num_inputs > 1, fcn.num_inputs
    assert fcn.num_inputs == project_2d.shape[0], (fcn.num_inputs, project_2d.shape)
    assert 0 <= fcn_num < fcn.num_outputs, (fcn_num, fcn.num_outputs)

    _center: np.ndarray = np.zeros(fcn.num_inputs) if center is None else center

    _center_orth_coor: np.ndarray = np.dot(_center, project_2d)  # (n,) x (n,2) = (2,)

    x: np.ndarray = np.linspace(xlim[0], xlim[1], NUM_PNTS_PER_AXIS_FOR_CONTOUR)
    y: np.ndarray = np.linspace(ylim[0], ylim[1], NUM_PNTS_PER_AXIS_FOR_CONTOUR)
    X, Y = np.meshgrid(x, y)

    Z = fcn.eval(
        np.dot(np.array([X, Y]).reshape((2, -1)).T - _center_orth_coor, project_2d.T) + _center
    ).reshape(X.shape)

    ax.contour(X, Y, Z, **kwargs)

    if eq_cnst_fcn is not None and isinstance(eq_cnst_fcn, AffineFunction):
        draw_2d_hyperplanes(
            ax, eq_cnst_fcn, project_2d, _center, xlim, ylim, "g-", alpha=0.1, zorder=-10
        )

    if ineq_cnst_fcn is not None and isinstance(ineq_cnst_fcn, AffineFunction):
        draw_2d_hyperplanes(
            ax, ineq_cnst_fcn, project_2d, _center, xlim, ylim, "b-", alpha=0.1, zorder=-10
        )


def draw_2d_hyperplanes(
    ax: Axes,
    affine_fcn: AffineFunction,
    prj_mat_2d: np.ndarray,
    center: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    /,
    *args,
    **kwargs,
) -> None:
    assert isinstance(affine_fcn, AffineFunction), affine_fcn.__class__
    center_orth_coor: np.ndarray = np.dot(center, prj_mat_2d)  # (n,) x (n,2) = (2,)

    for idx, b_scalar in enumerate(affine_fcn.b_array_1d):
        _a, _b, _c = np.dot(
            affine_fcn.a_array_2d[idx, :], np.hstack((prj_mat_2d, center[:, np.newaxis]))
        )
        draw_2d_line_in_box(
            ax,
            (_a, _b, _c + b_scalar - _a * center_orth_coor[0] - _b * center_orth_coor[1]),
            xlim,
            ylim,
            *args,
            **kwargs,
        )


def draw_2d_line_in_box(
    ax: Axes,
    coef: tuple[float, float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *args,
    **kwargs,
) -> None:
    x_list: list[float] = list()
    y_list: list[float] = list()

    a, b, c = coef
    if a != 0.0 and xlim[0] < -(c + b * ylim[0]) / a <= xlim[1]:
        x_list.append(-(c + b * ylim[0]) / a)
        y_list.append(ylim[0])

    if a != 0.0 and xlim[0] < -(c + b * ylim[1]) / a <= xlim[1]:
        x_list.append(-(c + b * ylim[1]) / a)
        y_list.append(ylim[1])

    if b != 0.0 and ylim[0] < -(c + a * xlim[0]) / b <= ylim[1]:
        x_list.append(xlim[0])
        y_list.append(-(c + a * xlim[0]) / b)

    if b != 0.0 and ylim[0] < -(c + a * xlim[1]) / b <= ylim[1]:
        x_list.append(xlim[1])
        y_list.append(-(c + a * xlim[1]) / b)

    assert len(x_list) == 0 or len(x_list) == 2, (x_list, y_list)

    if len(x_list) == 2:
        ax.plot(x_list, y_list, *args, **kwargs)
