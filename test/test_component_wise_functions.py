import unittest
from typing import List, Callable

from numpy import ndarray, allclose, exp, arctan
from numpy.random import randn

from functions.unit_functions import sigmoid
from functions.basic_functions.component_wise_function import ComponentWiseFunction


class TestComponentWiseFunctions(unittest.TestCase):
    def test_identity_component_wise_function(self):
        num_inputs: int = 100
        num_data: int = 1000

        identity_component_wise_function: ComponentWiseFunction = ComponentWiseFunction(lambda x: x)

        x_array_2d: ndarray = randn(num_data, num_inputs)
        y_array_2d: ndarray = identity_component_wise_function.get_y_values_2d(x_array_2d)

        self.assertTrue(allclose(y_array_2d, x_array_2d))

    def test_exp_component_wise_function(self):
        num_inputs: int = 100
        num_data: int = 1000

        exp_component_wise_function: ComponentWiseFunction = ComponentWiseFunction(exp)

        x_array_2d: ndarray = randn(num_data, num_inputs)
        y_array_2d: ndarray = exp_component_wise_function.get_y_values_2d(x_array_2d)

        self.assertTrue(allclose(y_array_2d, exp(x_array_2d)))

    def test_sigmoid_component_wise_function(self):
        num_inputs: int = 100
        num_data: int = 1000

        sigmoid_component_wise_function: ComponentWiseFunction = ComponentWiseFunction(sigmoid)

        x_array_2d: ndarray = randn(num_data, num_inputs)
        y_array_2d: ndarray = sigmoid_component_wise_function.get_y_values_2d(x_array_2d)

        self.assertTrue(allclose(y_array_2d, 1.0 / (1.0 + exp(-x_array_2d))))

    def test_generic_component_wise_function(self):
        ufcn_list: List[Callable] = [sigmoid, exp, arctan]
        num_inputs: int = len(ufcn_list)
        num_data: int = 1000

        component_wise_function: ComponentWiseFunction = ComponentWiseFunction(ufcn_list)

        x_array_2d: ndarray = randn(num_data, num_inputs)
        y_array_2d: ndarray = component_wise_function.get_y_values_2d(x_array_2d)

        for idx, ufcn in enumerate(ufcn_list):
            self.assertTrue(allclose(y_array_2d[:, idx], ufcn(x_array_2d[:, idx])))


if __name__ == "__main__":
    unittest.main()
