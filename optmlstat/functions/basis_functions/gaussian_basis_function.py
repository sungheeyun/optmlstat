"""

"""

from numpy import ndarray, array, vstack, zeros, exp
from numpy.linalg import inv

from optmlstat.functions.function_base import FunctionBase


class GaussianBasisFunction(FunctionBase):
    """
    Gaussian basis function.
    We do not multiply the scaling term as in Gaussian probability density function (PDF)
    since it can be dealt with coefficients.
    """

    def __init__(
        self,
        covariance_list: list[ndarray | float | int] | ndarray | float | int,
        mean_array_2d: ndarray | None = None,
        inverse: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        covariance_list:
          list of covariance matrices
        inverse:
          If inverse is True, covariance_list is interpreted
          as a list of the inverses of the covariance matrices.
        """
        if isinstance(covariance_list, int):
            covariance_list = float(covariance_list)

        if isinstance(covariance_list, float):
            covariance_list = array([[covariance_list]])

        if isinstance(covariance_list, ndarray):
            covariance_list = [covariance_list]

        assert len(covariance_list) > 0

        for idx, covariance in enumerate(covariance_list):
            if isinstance(covariance, (float, int)):
                covariance_list[idx] = array([[float(covariance)]])

        assert covariance_list[0].ndim == 2, covariance_list[0].ndim
        assert covariance_list[0].shape[0] == covariance_list[0].shape[1], covariance_list[0].shape
        for covariance in covariance_list:
            assert covariance_list[0].shape == covariance.shape, (
                covariance_list[0].shape,
                covariance.shape,
            )

        self.covariance_list: list[ndarray] = covariance_list

        self.mean_array_2d: ndarray = mean_array_2d

        if self.mean_array_2d is None:
            self.mean_array_2d = zeros((self.num_outputs, self.num_inputs))

        assert self.mean_array_2d.shape == (self.num_outputs, self.num_inputs)

        self.inverse_covariance_list: list[ndarray] = self.covariance_list
        if not inverse:
            self.inverse_covariance_list = [inv(covariance) for covariance in self.covariance_list]

    @property
    def num_inputs(self) -> int:
        return self.covariance_list[0].shape[0]

    @property
    def num_outputs(self) -> int:
        return len(self.covariance_list)

    @property
    def is_affine(self) -> bool:
        return False

    @property
    def is_strictly_convex(self) -> bool:
        return False

    @property
    def is_convex(self) -> bool:
        return False

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        y_array_1d_list: list[ndarray] = list()

        for idx, inverse_covariance in enumerate(self.inverse_covariance_list):
            x_array_2d_ = x_array_2d - self.mean_array_2d[idx, :]
            y_array_1d_list.append(
                exp(-0.5 * (x_array_2d_.dot(inverse_covariance) * x_array_2d_).sum(axis=1))
            )

        return vstack(y_array_1d_list).T
