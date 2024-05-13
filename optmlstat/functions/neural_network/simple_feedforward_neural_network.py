from typing import List, Callable

from numpy import ndarray

from optmlstat.functions.basic_functions.composite_function import CompositeFunction
from optmlstat.functions.basic_functions.component_wise_function import ComponentWiseFunction
from optmlstat.functions.basic_functions.affine_function import AffineFunction
from optmlstat.functions.function_base import FunctionBase


class SimpleFeedforwardNeuralNetwork(CompositeFunction):
    """
    A simple feed-forward neural network.
    """

    def __init__(self, weight_array_list: List[ndarray], activation_function: Callable) -> None:
        """
        If the number of hidden nodes of the previous layer is N
        and the number of hidden nodes of the next layer is M,
        then the corresponding weight_array should be (N + 1)-by-M array
        where the last row represents the bias terms.

        Parameters
        ----------
        weight_array_list:
         List of weight arrays.
        activation_function:
         Activation function
        """

        for idx, next_weight_array in enumerate(weight_array_list[1:]):
            prev_weight_array: ndarray = weight_array_list[idx]
            assert prev_weight_array.shape[1] + 1 == next_weight_array.shape[0]

        weight_affine_function_list: List[FunctionBase] = list()

        for idx, weight_array in enumerate(weight_array_list):
            if idx > 0:
                weight_affine_function_list.append(ComponentWiseFunction(activation_function))

            weight_affine_function_list.append(
                AffineFunction(weight_array[:-1, :], weight_array[-1, :])
            )

        super(SimpleFeedforwardNeuralNetwork, self).__init__(weight_affine_function_list)
