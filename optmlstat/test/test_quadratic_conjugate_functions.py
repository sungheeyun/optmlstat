import unittest
from logging import Logger, getLogger

from numpy import ndarray, newaxis, greater
from numpy.random import randn
from freq_used.logging_utils import set_logging_basic_config

from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction


logger: Logger = getLogger()


class TestQuadraticConjugateFunctions(unittest.TestCase):
    dim_of_domain: int = 30
    dim_of_range: int = 10
    num_data_points: int = 100
    TOLERANCE: float = 1e-6

    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_conjugate_of_quadratic_function(self):
        dim_of_domain: int = TestQuadraticConjugateFunctions.dim_of_domain
        dim_of_range: int = TestQuadraticConjugateFunctions.dim_of_range
        num_data_points: int = TestQuadraticConjugateFunctions.num_data_points

        p_array_3d: ndarray = ndarray((dim_of_domain, dim_of_domain, dim_of_range))

        for idx in range(dim_of_range):
            square_array_2d: ndarray = randn(dim_of_domain, dim_of_domain)
            p_array_2d: ndarray = square_array_2d.dot(square_array_2d.T)
            p_array_3d[:, :, idx] = p_array_2d
        q_array_2d: ndarray = randn(dim_of_domain, dim_of_range)
        r_array_1d: ndarray = randn(dim_of_range)

        quadratic_function: QuadraticFunction = QuadraticFunction(p_array_3d, q_array_2d, r_array_1d)
        conjugate_function: QuadraticFunction = quadratic_function.conjugate

        # x_array_2d: ndarray = randn(num_data_points, dim_of_domain)
        z_array_2d: ndarray = randn(num_data_points, dim_of_domain)

        conj_arg_x_array_3d: ndarray = quadratic_function.conjugate_arg(z_array_2d)
        logger.info(conj_arg_x_array_3d.shape)

        x_array_2d: ndarray = conj_arg_x_array_3d[:, :, 1]

        y_value_2d: ndarray = quadratic_function.get_y_values_2d(x_array_2d)
        conjugate_fcn_value_2d: ndarray = conjugate_function.get_y_values_2d(z_array_2d)

        logger.info(y_value_2d.shape)
        logger.info(conjugate_fcn_value_2d.shape)
        fcn_plus_conjugate_2d: ndarray = y_value_2d + conjugate_fcn_value_2d
        inner_product_of_x_and_y: ndarray = (x_array_2d * z_array_2d).sum(axis=1)

        self.assertEqual(fcn_plus_conjugate_2d.ndim, 2)
        self.assertEqual(fcn_plus_conjugate_2d.shape, (num_data_points, dim_of_range))
        self.assertEqual(inner_product_of_x_and_y.ndim, 1)
        self.assertEqual(inner_product_of_x_and_y.shape, (num_data_points,))

        logger.info(f"x_value: {x_array_2d}")
        logger.info(f"fnc_values: {y_value_2d}")
        logger.info(f"z_value: {z_array_2d}")
        logger.info(f"conjugate_fcn_values: {conjugate_fcn_value_2d}")
        logger.info(f"fcn + conjugate = {fcn_plus_conjugate_2d}")
        logger.info(f"z^T x = {inner_product_of_x_and_y}")
        logger.info(fcn_plus_conjugate_2d - inner_product_of_x_and_y[:, newaxis])
        logger.info((fcn_plus_conjugate_2d - inner_product_of_x_and_y[:, newaxis]).min())
        logger.info(greater(fcn_plus_conjugate_2d, inner_product_of_x_and_y[:, newaxis]))

        self.assertTrue(
            (
                fcn_plus_conjugate_2d - inner_product_of_x_and_y[:, newaxis]
                > -TestQuadraticConjugateFunctions.TOLERANCE
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
