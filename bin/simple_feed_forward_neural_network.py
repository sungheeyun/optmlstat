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

    num_inputs: int = 1
    num_hidden_nodes_1: int = 1000
    num_outputs: int = 1

    x_array_1d: ndarray = linspace(x_min, x_max, num_plotting_points)

    figure: Figure
    axes: Axes
    figure, axes = plt.subplots()
    for idx in range(num_plots):
        weight_array_list: List[ndarray] = [
            10 * randn(num_inputs + 1, num_hidden_nodes_1),
            randn(num_hidden_nodes_1 + 1, num_outputs),
        ]
        two_layer_feed_forward_neural_network: SimpleFeedforwardNeuralNetwork = SimpleFeedforwardNeuralNetwork(
            weight_array_list, sigmoid
        )
        y_array_1d: ndarray = two_layer_feed_forward_neural_network.get_y_values_2d_from_x_values_1d(x_array_1d).ravel()
        axes.plot(x_array_1d, y_array_1d)

    figure.show()

    if "__file__" in dir():
        plt.show()
