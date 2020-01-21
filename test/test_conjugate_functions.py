import unittest
from logging import Logger, getLogger

from numpy import ndarray, newaxis, array
from numpy.random import randn
from freq_used.logging import set_logging_basic_config

from functions.basic_functions.quadratic_function import QuadraticFunction


logger: Logger = getLogger()


class TestConjugateFunctions(unittest.TestCase):
    dim_of_domain: int = 10
    num_data_points: int = 100000

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_conjugate_of_quadratic_function(self):
        dim_of_domain: int = TestConjugateFunctions.dim_of_domain
        num_data_points: int = TestConjugateFunctions.num_data_points

        square_array_2d: ndarray = randn(dim_of_domain, dim_of_domain)
        p_array_2d: ndarray = square_array_2d.dot(square_array_2d.T)
        q_array_1d: ndarray = randn(dim_of_domain)
        r_scalar: float = randn()

        quadratic_function: QuadraticFunction = QuadraticFunction(
            p_array_2d[:, :, newaxis], q_array_1d[:, newaxis], array([r_scalar], float)
        )

        conjugate_function: QuadraticFunction = quadratic_function.conjugate

        x_array_2d: ndarray = randn(num_data_points, dim_of_domain)
        y_array_2d: ndarray = randn(num_data_points, dim_of_domain)

        fcn_value_2d: ndarray = quadratic_function.get_y_values_2d(x_array_2d)
        conjugate_fcn_value_2d: ndarray = conjugate_function.get_y_values_2d(y_array_2d)

        fcn_plus_conjugate: ndarray = (fcn_value_2d + conjugate_fcn_value_2d).ravel()
        inner_product_of_x_and_y: ndarray = (x_array_2d * y_array_2d).sum(axis=1)

        logger.info(f"x_value: {x_array_2d}")
        logger.info(f"fnc_values: {fcn_value_2d}")
        logger.info(f"y_value: {y_array_2d}")
        logger.info(f"conjugate_fcn_values: {conjugate_fcn_value_2d}")
        logger.info(f"fcn + conjugate = {fcn_plus_conjugate}")
        logger.info(f"x^y = {inner_product_of_x_and_y}")
        logger.info((fcn_plus_conjugate - inner_product_of_x_and_y).min())

        self.assertTrue((fcn_plus_conjugate >= inner_product_of_x_and_y).all())


if __name__ == "__main__":
    unittest.main()
