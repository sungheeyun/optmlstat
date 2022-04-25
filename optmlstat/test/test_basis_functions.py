import unittest
from logging import Logger, getLogger

from numpy import ndarray, linspace, array
from freq_used.logging_utils import set_logging_basic_config
from freq_used.plotting import get_figure
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from optmlstat.functions.basis_functions.gaussian_basis_function import (
    GaussianBasisFunction,
)

logger: Logger = getLogger()


class TestBasisFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)

    def test_gaussian_basis_functions(self):
        num_outputs: int = 10
        gaussian_basis_function: GaussianBasisFunction = GaussianBasisFunction(
            [4] * num_outputs,
            array([linspace(-3, 3, num_outputs)]).T,
            inverse=True,
        )
        logger.info(gaussian_basis_function)
        logger.info(gaussian_basis_function.covariance_list)
        logger.info(gaussian_basis_function.num_inputs)
        logger.info(gaussian_basis_function.num_outputs)

        N: int = 100

        x_array_2d: ndarray = array([linspace(-10.0, 10.0, N)]).T
        y_array_2d: ndarray = gaussian_basis_function.get_y_values_2d(
            x_array_2d
        )

        fig: Figure = get_figure(1, 1)
        ax: Axes = fig.get_axes()[0]
        for idx in range(num_outputs):
            ax.plot(x_array_2d.ravel(), y_array_2d[:, idx])
        fig.show()

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
