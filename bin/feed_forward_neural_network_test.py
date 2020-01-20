from typing import List
from logging import Logger, getLogger

from numpy import ndarray, linspace
from numpy.random import randn
from freq_used.logging import set_logging_basic_config
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from functions.unit_functions import sigmoid
from functions.neural_network.simple_feedforward_neural_network import SimpleFeedforwardNeuralNetwork


logger: Logger = getLogger()

if __name__ == "__main__":
    set_logging_basic_config(__file__)

    x_min: float = -10.0
    x_max: float = 10.0
    num_plotting_points: int = 100
    num_plots: int = 10

    dim_list: List[int] = [1, 1000, 100, 1]

    x_array_1d: ndarray = linspace(x_min, x_max, num_plotting_points)

    figure: Figure
    axes: Axes
    figure, axes = plt.subplots()
    for idx in range(num_plots):
        weight_array_list: List[ndarray] = [
            5.0 * randn(dim_list[i] + 1, dim_list[i + 1]) for i in range(len(dim_list) - 1)
        ]
        simple_feed_forward_neural_network: SimpleFeedforwardNeuralNetwork = SimpleFeedforwardNeuralNetwork(
            weight_array_list, sigmoid
        )
        logger.info(simple_feed_forward_neural_network.get_shape_tuple())
        y_array_1d: ndarray = simple_feed_forward_neural_network.get_y_values_2d_from_x_values_1d(x_array_1d).ravel()
        axes.plot(x_array_1d, y_array_1d)

    figure.show()

    if "__file__" in dir():
        plt.show()
