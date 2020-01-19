from logging import Logger, getLogger

from numpy import ndarray, linspace
from numpy.random import randn
from freq_used.logging import set_logging_basic_config
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from optmlstat.functions.unit_functions import sigmoid
from optmlstat.functions.component_wise_function import ComponentWiseFunction
from optmlstat.functions.composite_function import CompositeFunction
from optmlstat.functions.affine_function import AffineFunction


logger: Logger = getLogger()


if __name__ == "__main__":
    set_logging_basic_config(__file__)

    x_min: float = -10.0
    x_max: float = 10.0
    num_plotting_points: int = 100

    num_hidden_nodes: int = 1000

    weight_function_1: AffineFunction = AffineFunction(randn(1, num_hidden_nodes), randn(num_hidden_nodes))
    weight_function_2: AffineFunction = AffineFunction(randn(num_hidden_nodes, 1), randn(1))
    activation_function: ComponentWiseFunction = ComponentWiseFunction(sigmoid)

    logger.info(weight_function_1.get_num_inputs())
    logger.info(weight_function_1.get_num_outputs())
    logger.info(weight_function_2.get_num_inputs())
    logger.info(weight_function_2.get_num_outputs())
    logger.info(activation_function.get_num_inputs())
    logger.info(activation_function.get_num_outputs())

    two_layer_feed_forward_neural_network: CompositeFunction = CompositeFunction(
        [weight_function_1, activation_function, weight_function_2]
    )

    x_array_1d: ndarray = linspace(x_min, x_max, num_plotting_points)
    y_array_1d: ndarray = two_layer_feed_forward_neural_network.get_y_values_1d(x_array_1d)

    figure: Figure
    axes: Axes
    figure, axes = plt.subplots()
    axes.plot(x_array_1d, y_array_1d)
    figure.show()
