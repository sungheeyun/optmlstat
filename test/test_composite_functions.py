import unittest

from numpy import ndarray, allclose
from numpy.random import randn

from optmlstat.functions.composite_function import CompositeFunction
from optmlstat.functions.affine_function import AffineFunction


class TestCompositeFunctions(unittest.TestCase):
    def test_simple_composite_function(self) -> None:
        num_data: int = 1000
        num_variables_1: int = 100
        num_variables_2: int = 10
        num_variables_3: int = 30

        slope_array_1: ndarray = randn(num_variables_1, num_variables_2)
        intercept_array_1: ndarray = randn(num_variables_2)

        slope_array_2: ndarray = randn(num_variables_2, num_variables_3)
        intercept_array_2: ndarray = randn(num_variables_3)

        affine_function_1: AffineFunction = AffineFunction(slope_array_1, intercept_array_1)
        affine_function_2: AffineFunction = AffineFunction(slope_array_2, intercept_array_2)

        composite_function: CompositeFunction = CompositeFunction([affine_function_1, affine_function_2])

        x_array_2d: ndarray = randn(num_data, num_variables_1)

        y_array_2d_1: ndarray = affine_function_1.get_y_values_2d(x_array_2d)
        y_array_2d_2: ndarray = affine_function_2.get_y_values_2d(y_array_2d_1)
        y_array_2d_3: ndarray = composite_function.get_y_values_2d(x_array_2d)

        self.assertTrue(allclose(CompositeFunction([]).get_y_values_2d(x_array_2d), x_array_2d))
        self.assertTrue(allclose(CompositeFunction([affine_function_1]).get_y_values_2d(x_array_2d), y_array_2d_1))
        self.assertTrue(allclose(y_array_2d_3, y_array_2d_2))

    def test_num_dimensions(self) -> None:

        self.assertIsNone(CompositeFunction([]).get_num_inputs())
        self.assertIsNone(CompositeFunction([]).get_num_outputs())

        affine_function_1: AffineFunction = AffineFunction(randn(3, 2), randn(2))
        affine_function_2: AffineFunction = AffineFunction(randn(2, 10), randn(10))

        self.assertEqual(affine_function_1.get_num_inputs(), 3)
        self.assertEqual(affine_function_1.get_num_outputs(), 2)
        self.assertEqual(affine_function_2.get_num_inputs(), 2)
        self.assertEqual(affine_function_2.get_num_outputs(), 10)

        self.assertIsNone(CompositeFunction([]).get_num_inputs())
        self.assertIsNone(CompositeFunction([]).get_num_outputs())
        self.assertEqual(CompositeFunction([affine_function_1]).get_num_inputs(), 3)
        self.assertEqual(CompositeFunction([affine_function_1]).get_num_outputs(), 2)
        self.assertEqual(CompositeFunction([affine_function_1, affine_function_2]).get_num_inputs(), 3)
        self.assertEqual(CompositeFunction([affine_function_1, affine_function_2]).get_num_outputs(), 10)


if __name__ == "__main__":
    unittest.main()
