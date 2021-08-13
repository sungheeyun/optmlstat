from typing import Tuple
import abc

import numpy as np

from functions.function_base import FunctionBase
from ml.modeling.bayesian_modeler_base import BayesianModelerBase


class BayesianLeastSquaresBase(BayesianModelerBase):
    def get_predictor(self) -> FunctionBase:
        assert False

    @abc.abstractmethod
    def get_predictive_dist(
        self, x_array_1d: np.ndarray
    ) -> Tuple[float, float]:
        pass

    @classmethod
    def solve_linear_sys_using_lower_tri_from_chol_fac(
        cls, lower_tri: np.ndarray, y_array_1d: np.ndarray
    ) -> np.ndarray:
        z_array_1d: np.ndarray = cls.forward_substitution(
            lower_tri, y_array_1d
        )
        x_array_1d: np.ndarray = cls.backward_substitution(
            lower_tri.T, z_array_1d
        )
        return x_array_1d

    @classmethod
    def forward_substitution(
        cls, lower_tri: np.ndarray, y_array_1d: np.ndarray
    ) -> np.ndarray:
        vec_size: int = y_array_1d.size
        assert lower_tri.shape == (vec_size, vec_size), (
            lower_tri.shape,
            y_array_1d.shape,
        )

        x_array_1d: np.ndarray = np.ndarray(shape=(vec_size,), dtype=float)

        for idx in range(vec_size):
            x_array_1d[idx] = (
                y_array_1d[idx]
                - np.dot(lower_tri[idx, :idx], x_array_1d[:idx])
            ) / lower_tri[idx, idx]

        return x_array_1d

    @classmethod
    def backward_substitution(
        cls, upper_tri: np.ndarray, y_array_1d: np.ndarray
    ) -> np.ndarray:
        vec_size: int = y_array_1d.size
        assert upper_tri.shape == (vec_size, vec_size), (
            upper_tri.shape,
            y_array_1d.shape,
        )

        x_array_1d: np.ndarray = np.ndarray(shape=(vec_size,), dtype=float)

        for idx in range(vec_size - 1, -1, -1):
            x_array_1d[idx] = (
                y_array_1d[idx]
                - np.dot(upper_tri[idx, idx + 1:], x_array_1d[idx + 1:])
            ) / upper_tri[idx, idx]

        return x_array_1d
