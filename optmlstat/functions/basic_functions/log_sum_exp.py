"""
sum exp of linear function
sum exp (Ax+b)
"""

import numpy as np

from optmlstat.functions.fcn_decorators import fcn_evaluator
from optmlstat.functions.function_base import FunctionBase


class LogSumExp(FunctionBase):

    def __init__(self, A_3d: np.ndarray | list, b_2d: np.ndarray | list) -> None:
        """
        :param A_3d:
            A_3d[i,:,:] = A of i-th function
        :param b_2d:
            b_2d[i,:] = b of i-th function
        """

        self.A_3d: np.ndarray = np.array(A_3d, float)
        self.b_2d: np.ndarray = np.array(b_2d, float)

        assert self.A_3d.ndim == 3, self.A_3d.shape
        assert self.b_2d.ndim == 2, self.b_2d.shape
        assert self.A_3d.shape[0] == self.b_2d.shape[0], (
            self.A_3d.shape,
            self.b_2d.shape,
        )  # # functions
        assert self.A_3d.shape[1] == self.b_2d.shape[1], (
            self.A_3d.shape,
            self.b_2d.shape,
        )  # # terms

    @property
    def num_inputs(self) -> int:
        return self.A_3d.shape[-1]

    @property
    def num_outputs(self) -> int:
        return self.A_3d.shape[0]

    @property
    def is_affine(self) -> bool:
        return False

    @property
    def is_strictly_convex(self) -> bool:
        # however, is strictly convex if every function depends on every variable
        return False

    @property
    def is_convex(self) -> bool:
        return True

    @property
    def is_strictly_concave(self) -> bool:
        return False

    @property
    def is_concave(self) -> bool:
        return False

    @property
    def is_differentiable(self) -> bool:
        return True

    @fcn_evaluator
    def get_y_values_2d(self, x_array_2d: np.ndarray) -> np.ndarray:
        y_2d: np.ndarray = np.zeros((x_array_2d.shape[0], self.num_outputs))
        for idx, A_2d in enumerate(self.A_3d):  # for each function
            y_2d[:, idx] = np.exp(np.dot(x_array_2d, A_2d.T) + self.b_2d[idx]).sum(axis=1)

        return np.log(y_2d)

    @fcn_evaluator
    def jacobian(self, x_array_2d: np.ndarray) -> np.ndarray:
        jac_3d: np.ndarray = np.zeros((x_array_2d.shape[0], self.num_outputs, self.num_inputs))
        for idx, A_2d in enumerate(self.A_3d):  # for each function
            jac_3d[:, idx, :] = np.dot(np.exp(np.dot(x_array_2d, A_2d.T) + self.b_2d[idx]), A_2d)
        y_2d: np.ndarray = self.get_y_values_2d(x_array_2d)
        jac_3d /= np.exp(y_2d)[:, :, None]

        return jac_3d

    @property
    def conjugate(self) -> FunctionBase:
        raise NotImplementedError()

    def conjugate_arg(self, z_array_2d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
