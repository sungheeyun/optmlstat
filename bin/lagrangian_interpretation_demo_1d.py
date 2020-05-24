from typing import List, Tuple, Dict, Any, Callable, Optional, Iterable
from logging import Logger, getLogger
from enum import Enum

from numpy import ndarray, linspace, power, ones, zeros, zeros_like
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
from freq_used.logging import set_logging_basic_config
from freq_used.plotting import get_figure

from optmlstat.functions.function_base import FunctionBase
from optmlstat.functions.basic_functions.affine_function import AffineFunction
from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction


logger: Logger = getLogger()
mpl.use("TkAgg")


LAGRANGE_MULTIPLIER_SLIDER_WIDTH: float = 0.2
LAGRANGE_MULTIPLIER_SLIDER_HEIGHT: float = 0.03
LAGRANGE_MULTIPLIER_CENTER_POSITION: float = 0.3
LAGRANGE_MULTIPLIER_SLIDER_BOTTOM_POSITION: float = 0.90
LAGRANGE_MULTIPLIER_SLIDER_COLOR: str = "lightgoldenrodyellow"

RESET_BUTTON_WIDTH: float = 0.1
RESET_BUTTON_HEIGHT: float = 0.03
RESET_BUTTON_CENTER_POSITION: float = 0.6
RESET_BUTTON_BOTTOM_POSITION: float = 0.90


class ButtonSliderBase:
    """
    Base class for Button or Slider class.
    """

    def __init__(
        self, figure: Figure, width: float, height: float, center_position: float, bottom_position: float
    ) -> None:
        self.figure: Figure = figure
        self.width: float = width
        self.height: float = height
        self.center_position: float = center_position
        self.bottom_position: float = bottom_position

        self.axis: Axes = figure.add_axes(
            [self.center_position - self.width / 2.0, self.bottom_position, self.width, self.height]
        )


class MySlider(ButtonSliderBase):
    """
    Wrapper class for matplotlib.widgets.Slider.
    """

    def __init__(
        self,
        figure: Figure,
        width: float,
        height: float,
        center_position: float,
        bottom_position: float,
        slider_kwargs: Dict[str, Any],
    ) -> None:
        super(MySlider, self).__init__(figure, width, height, center_position, bottom_position)

        self.slider: Slider = Slider(ax=self.axis, **slider_kwargs)


class MyButton(ButtonSliderBase):
    """
    Wrapper class for matplotlib.widgets.Slider.
    """

    def __init__(
        self,
        figure: Figure,
        width: float,
        height: float,
        center_position: float,
        bottom_position: float,
        button_kwargs: Dict[str, Any],
    ) -> None:
        super(MyButton, self).__init__(figure, width, height, center_position, bottom_position)

        self.button: Button = Button(ax=self.axis, **button_kwargs)


class LagrangianIllustrator:
    """
    This class shows graphically how Lagrange multipliers work
    to help clients obtain geometrical understanding of their role in constrained optimization.

    Consider the following (one-dimensional) equality constrained optimization problem:

        minimize f(x)
        subject to h(x) >= 0 or h(x) = 0 or h(x) <= 0

    where f: R -> R and h: R -> R are the objective function and constraint function respectively.
    Depending on the type of constraint, the interpretation of the Lagrange multiplier changes.

    The Lagrangian is defined by

        L(x, lambda) = f(x) + lambda h(x)

    The Lagrange dual function is defined by

        g(lambda) = inf_x L(x, lambda)

    If the constraint in the above problem is ">=", then lambda <=0,
    and if the constraint is "<=", then lambda >= 0,
    and if the constraint is "=", then lambda can take any real values.
    """

    NUM_PLOTTING_POINTS: int = 100
    ALPHA: float = 0.5

    def __init__(
        self,
        obj_fcn: FunctionBase,
        const_fcn: FunctionBase,
        cnst_boundary_points: Iterable[float],
        minimum_point_fcn: Optional[Callable] = None,
        minimum_value_fcn: Optional[Callable] = None,
    ) -> None:
        self.obj_fcn: FunctionBase = obj_fcn
        self.const_fcn: FunctionBase = const_fcn
        self.cnst_boundary_points: List[float] = list(cnst_boundary_points)

        self.minimum_point_fcn: Callable = lambda x: 0.0
        if minimum_point_fcn is not None:
            self.minimum_point_fcn = minimum_point_fcn

        self.minimum_value_fcn: Callable = lambda x: 0.0
        if minimum_value_fcn is not None:
            self.minimum_value_fcn = minimum_value_fcn

    def create_interactive_plot(
        self, x_min: float, x_max: float, lambda_min: float, lambda_max: float, initial_lambda: float
    ) -> Figure:

        x_array_1d: ndarray = linspace(x_min, x_max, LagrangianIllustrator.NUM_PLOTTING_POINTS)
        obj_fcn_array_1d: ndarray = self.obj_fcn.get_y_values_2d_from_x_values_1d(x_array_1d).ravel()
        cnst_fcn_array_1d: ndarray = self.const_fcn.get_y_values_2d_from_x_values_1d(x_array_1d).ravel()

        figure: Figure = get_figure(
            1, 1, axis_width=4, axis_height=3, left_margin=0.5, right_margin=3.0, bottom_margin=0.5
        )
        axis: Axes = figure.get_axes()[0]

        axis.plot(x_array_1d, obj_fcn_array_1d, "k-", label="objective function", alpha=LagrangianIllustrator.ALPHA)
        axis.plot(
            x_array_1d, cnst_fcn_array_1d, "b-", label="equality constraint function", alpha=LagrangianIllustrator.ALPHA
        )
        ylim: Tuple[float] = axis.get_ylim()
        for cnst_boundary_point in self.cnst_boundary_points:
            axis.plot(
                ones(2) * cnst_boundary_point,
                ylim,
                "b-.",
                label="constraint boundary",
                alpha=LagrangianIllustrator.ALPHA,
            )

        lagrangian_line_2d_list: List[Line2D] = axis.plot(x_array_1d, zeros_like(x_array_1d), "r-", label="Lagrangian")
        lagrangian_minimum_x_list_2d_list = axis.plot(zeros(2), ylim, "r-.", label="Lagrangian minimum x")
        lagrangian_minimum_y_line_2d_list: List[Line2D] = axis.plot(
            [x_min, x_max], zeros(2), "r-.", label="Lagrangian minimum y"
        )

        obj_minimum_point_line_2d_list: List[Line2D] = axis.plot([0.0], [0.0], "o", markersize=8)
        lagrangian_minimum_point_line_2d_list: List[Line2D] = axis.plot(
            [0.0], [0.0], "o", markersize=8, markerfacecolor="r", markeredgecolor="none"
        )

        axis.legend(bbox_to_anchor=(1.05, 0.9))

        assert len(lagrangian_line_2d_list) == 1
        assert len(lagrangian_minimum_y_line_2d_list) == 1
        assert len(lagrangian_minimum_x_list_2d_list) == 1
        assert len(obj_minimum_point_line_2d_list) == 1
        assert len(lagrangian_minimum_point_line_2d_list) == 1

        lagrangian_line_2d: Line2D = lagrangian_line_2d_list[0]
        lagrangian_minimum_x_list_2d: Line2D = lagrangian_minimum_x_list_2d_list[0]
        lagrangian_minimum_y_line_2d: Line2D = lagrangian_minimum_y_line_2d_list[0]
        lagrangian_minimum_point_line_2d: Line2D = lagrangian_minimum_point_line_2d_list[0]

        obj_minimum_point_line_2d: Line2D = obj_minimum_point_line_2d_list[0]

        def update_lagrangian_in_x(value: float):
            lagrangian_minimum_x: float = self.minimum_point_fcn(value)
            lagrangian_minimum_y: float = self.minimum_value_fcn(value)

            lagrangian_line_2d.set_ydata(obj_fcn_array_1d + value * cnst_fcn_array_1d)
            lagrangian_minimum_x_list_2d.set_xdata(ones(2) * lagrangian_minimum_x)
            lagrangian_minimum_y_line_2d.set_ydata(lagrangian_minimum_y * ones(2))
            lagrangian_minimum_point_line_2d.set_xdata(ones(1) * lagrangian_minimum_x)
            lagrangian_minimum_point_line_2d.set_ydata(ones(1) * lagrangian_minimum_y)

            obj_minimum_point_line_2d.set_xdata(ones(1) * lagrangian_minimum_x)
            obj_minimum_point_line_2d.set_ydata(
                ones(1) * obj_fcn.get_y_values_2d_from_x_values_1d(ones(1) * lagrangian_minimum_x).ravel()
            )
            if self.const_fcn.get_y_values_2d_from_x_values_1d(ones(1) * lagrangian_minimum_x)[0, 0] <= 0.0:
                obj_minimum_point_line_2d.set_color("b")
            else:
                obj_minimum_point_line_2d.set_color("r")

            figure.canvas.draw_idle()

        lagrange_multiplier_slider: MySlider = MySlider(
            figure,
            LAGRANGE_MULTIPLIER_SLIDER_WIDTH,
            LAGRANGE_MULTIPLIER_SLIDER_HEIGHT,
            LAGRANGE_MULTIPLIER_CENTER_POSITION,
            LAGRANGE_MULTIPLIER_SLIDER_BOTTOM_POSITION,
            dict(label="Lagrange multiplier", valmin=lambda_min, valmax=lambda_max, valinit=initial_lambda),
        )
        lagrange_multiplier_slider.slider.on_changed(update_lagrangian_in_x)

        def reset(event):
            lagrange_multiplier_slider.slider.reset()

        reset_button: MyButton = MyButton(
            figure,
            RESET_BUTTON_WIDTH,
            RESET_BUTTON_HEIGHT,
            RESET_BUTTON_CENTER_POSITION,
            RESET_BUTTON_BOTTOM_POSITION,
            dict(label="Reset"),
        )
        reset_button.button.on_clicked(reset)

        update_lagrangian_in_x(initial_lambda)

        plt.show()

        return figure


class Problem(Enum):
    LCQM: int = 10  # linearly constrained quadratic program
    QCLM: int = 20  # quadratically constrained linear program


if __name__ == "__main__":

    set_logging_basic_config(__file__)

    case: Problem
    case_num: int = int(input("case? "))
    if case_num == 1:
        case = Problem.LCQM
    elif case_num == 2:
        case = Problem.QCLM
    else:
        assert False, case_num

    obj_fcn: FunctionBase
    cnst_fcn: FunctionBase
    cnst_boundary_points: List[float]

    if case == Problem.LCQM:
        """
        f(x) = x^2
        h(x) = x - 1
        """
        obj_fcn = QuadraticFunction(ones((1, 1, 1)), zeros((1, 1)), zeros(1))
        cnst_fcn = AffineFunction(ones((1, 1)), -ones(1))

        cnst_boundary_points = [1.0]

        def minimum_of_lagrangian(nu: float) -> float:
            return -power(nu, 2.0) / 4.0 - nu

        def minimum_point(nu: float) -> float:
            return -nu / 2.0

    elif case == Problem.QCLM:
        """
            f(x) = x
            h(x) = x^2 - 1

            L(x, lambda) = x + lambda (x^2 - 1)
            d L(x, lambda) / dx = 1 + 2 lambda x

            g(lambda) = - 1 / 2lambda + lambda(1/4lambda^2 - 1) = - 1 / 4lambda - lambda
        """
        obj_fcn = AffineFunction(ones((1, 1)), zeros(1))
        cnst_fcn = QuadraticFunction(ones((1, 1, 1)), zeros((1, 1)), -ones(1))

        cnst_boundary_points = [-1.0, 1.0]

        def minimum_of_lagrangian(nu: float) -> float:
            return -nu - 1.0 / (4 * abs(nu) + 1e-6) * (1.0 if nu >= 0.0 else -1.0)

        def minimum_point(nu: float) -> float:
            return -1.0 / (2.0 * abs(nu) + 1e-6) * (1.0 if nu >= 0.0 else -1.0)

    else:
        assert False, case

    lambda_min: float = -8.0
    lambda_max: float = 8.0
    initial_lambda: float = 0.0

    x_min: float = -3.0
    x_max: float = 3.0

    lagrangian_illustrator: LagrangianIllustrator = LagrangianIllustrator(
        obj_fcn, cnst_fcn, cnst_boundary_points, minimum_point, minimum_of_lagrangian
    )
    lagrangian_illustrator.create_interactive_plot(x_min, x_max, lambda_min, lambda_max, initial_lambda)
