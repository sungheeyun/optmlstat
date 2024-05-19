"""
quadratic functions
"""

from __future__ import annotations

from logging import Logger, getLogger

import numpy as np
from numpy import ndarray, vstack, array, stack
from scipy import linalg as la

from optmlstat.functions.function_base import FunctionBase
from optmlstat.functions.exceptions import (
    UnboundedBelowException,
    InfiniteNumberOfSolutionsException,
)

logger: Logger = getLogger()


class QuadraticFunction(FunctionBase):
    """
    quadratic function
    """

    @property
    def maximal_point(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def maximal_value(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def maximum_point(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def maximum_value(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def minimal_point(self) -> np.ndarray:
        return self.minimum_point[np.newaxis, :]

    @property
    def minimal_value(self) -> np.ndarray:
        if self.num_outputs != 1:
            raise NotImplementedError()

        return self.minimum_value[np.newaxis, :]

    @property
    def minimum_point(self) -> np.ndarray:
        if self.num_outputs != 1:
            raise NotImplementedError()

        if self.quad_3d is None:
            if np.abs(self.slope_2d[:, 0]).sum() == 0.0:
                raise InfiniteNumberOfSolutionsException()
            else:
                raise UnboundedBelowException()

        quad_array_2d: np.ndarray = self.quad_3d[:, :, 0]
        slope_array_1d: np.ndarray = self.slope_2d[:, 0]

        if np.any(la.eig(quad_array_2d)[0] < 0):
            raise UnboundedBelowException()

        sol: np.ndarray = la.lstsq(2 * quad_array_2d, -slope_array_1d)[0]

        if not np.allclose(np.dot(2 * quad_array_2d, sol), -slope_array_1d):
            raise UnboundedBelowException()

        return sol

    @property
    def minimum_value(self) -> np.ndarray:
        if self.num_outputs != 1:
            raise NotImplementedError()

        if self.quad_3d is None:
            if np.abs(self.slope_2d[:, 0]).sum() == 0.0:
                return self.intercept_1d[0]
            else:
                return np.array([-np.inf])

        quad_array_2d: np.ndarray = self.quad_3d[:, :, 0]
        slope_array_1d: np.ndarray = self.slope_2d[:, 0]
        intercept: float = float(self.intercept_1d[0])

        if np.any(la.eig(quad_array_2d)[0] < 0):
            return np.array([-np.inf])

        sol: np.ndarray = la.lstsq(2 * quad_array_2d, -slope_array_1d)[0]

        if not np.allclose(np.dot(2 * quad_array_2d, sol), -slope_array_1d):
            return np.array([-np.inf])

        return np.array([np.dot(sol, slope_array_1d) / 2.0 + intercept])

    @staticmethod
    def test_convexity(quad_array_3d: ndarray) -> tuple[bool, bool]:
        is_strictly_convex: bool = True
        is_convex: bool = True
        for idx3 in range(quad_array_3d.shape[2]):
            symmetric_array: ndarray = quad_array_3d[:, :, idx3] + quad_array_3d[:, :, idx3].T
            eigen_value_array: ndarray = la.eig(symmetric_array)[0]
            if (eigen_value_array <= 0.0).any():
                is_strictly_convex = False

            if (eigen_value_array < 0.0).any():
                is_convex = False

        return is_convex, is_strictly_convex

    def __init__(
        self,
        quad_array_3d: ndarray | None,
        slope_array_2d: ndarray,
        intercept_array_1d: ndarray,
    ) -> None:
        """
        If n is the number of inputs and m is that of outputs,
        quad_3d[:, :, i], slope_2d[:, i] and
        intercept_1d[i] represents :math:`P`, :math:`q`, and :math:`r` respectively
        in the following equation:

          :math:`f_i(x) = x^T P x + q^T x + r`

        Parameters
        ----------
        quad_array_3d:
         n-by-n-by-m ndarray
        slope_array_2d:
         n-by-m ndarray
        intercept_array_1d: m
         m ndarray
        """
        # check dimensions
        assert quad_array_3d is None or quad_array_3d.ndim == 3, quad_array_3d.ndim
        assert slope_array_2d.ndim == 2, slope_array_2d.ndim
        assert intercept_array_1d.ndim == 1, intercept_array_1d.ndim

        # check number of inputs
        assert quad_array_3d is None or quad_array_3d.shape[0] == slope_array_2d.shape[0]
        assert quad_array_3d is None or quad_array_3d.shape[1] == slope_array_2d.shape[0]

        # check number of outputs
        assert slope_array_2d.shape[1] == intercept_array_1d.size, (
            slope_array_2d.shape,
            intercept_array_1d.shape,
        )
        assert quad_array_3d is None or quad_array_3d.shape[2] == intercept_array_1d.size, (
            slope_array_2d.shape,
            intercept_array_1d.shape,
        )

        self.quad_3d: ndarray | None = None if quad_array_3d is None else quad_array_3d.copy()
        self.slope_2d: ndarray = slope_array_2d.copy()
        self.intercept_1d: ndarray = intercept_array_1d.copy()

        self._is_affine: bool = True if self.quad_3d is None else (self.quad_3d == 0.0).all()
        self._is_convex: bool
        self._is_strictly_convex: bool
        self._is_concave: bool
        self._is_strictly_concave: bool

        if self.quad_3d is None:
            self._is_convex = self._is_concave = True
            self._is_strictly_convex = self._is_strictly_concave = True
        else:
            (
                self._is_convex,
                self._is_strictly_convex,
            ) = self.test_convexity(self.quad_3d)
            (
                self._is_concave,
                self._is_strictly_concave,
            ) = self.test_convexity(-self.quad_3d)

    @property
    def num_inputs(self) -> int:
        return self.slope_2d.shape[0]

    @property
    def num_outputs(self) -> int:
        return self.slope_2d.shape[1]

    @property
    def is_affine(self) -> bool:
        return self._is_affine

    @property
    def is_convex(self) -> bool:
        return self._is_convex

    @property
    def is_strictly_convex(self) -> bool:
        return self._is_strictly_convex

    @property
    def is_concave(self) -> bool:
        return self._is_concave

    @property
    def is_strictly_concave(self) -> bool:
        return self._is_strictly_concave

    @property
    def is_differentiable(self) -> bool:
        return True

    @property
    def is_twice_differentiable(self) -> bool:
        return True

    @property
    def conjugate(self) -> QuadraticFunction:
        """
        If the (quadratic) function is not convex, this value can be infinity. (not always though)
        If the function is convex, but not strictly convex, the conjugate is bounded,
         but the calculation is a bit complicated.
        For now, we implement this function only when the function is strictly convex.

        In this case, when :math:`f(x) = x^T P x + q^T x + r`,
        the conjugate is :math:`(z^T P^{-1} z - 2 q^T P^{-1} z + q^T P^{-1} q)/4 - r`.
        """
        assert self.is_strictly_convex
        assert self.quad_3d is not None

        conjugate_quad_array_3d: ndarray = ndarray(shape=self.quad_3d.shape, dtype=float)
        conjugate_slope_array_1d_list: list[ndarray] = list()
        conjugate_intercept_list: list[float] = list()

        for idx in range(self.quad_3d.shape[2]):
            p_array_2d: ndarray = self.quad_3d[:, :, idx]
            conjugate_quad_array_3d[:, :, idx] = 0.25 * la.inv(p_array_2d)

            q_array_1d: ndarray = self.slope_2d[:, idx]
            p_inv_q_array_1d: ndarray = la.solve(p_array_2d, q_array_1d)
            conjugate_slope_array_1d_list.append(-0.5 * p_inv_q_array_1d)

            conjugate_intercept_list.append(0.25 * q_array_1d.dot(p_inv_q_array_1d))

        conjugate_slope_array_2d: ndarray = vstack(conjugate_slope_array_1d_list).T
        conjugate_intercept_array_1d: ndarray = (
            array(conjugate_intercept_list, float) - self.intercept_1d
        )

        logger.debug(f"conjugate_slope_array_1d_list: {conjugate_slope_array_1d_list}")
        logger.debug(f"conjugate_slope_array_2d.shape: {conjugate_slope_array_2d.shape}")

        return QuadraticFunction(
            conjugate_quad_array_3d,
            conjugate_slope_array_2d,
            conjugate_intercept_array_1d,
        )

    def conjugate_arg(self, z_array_2d: ndarray) -> ndarray:
        """
        The gradient of :math:`z^T x - f(x) = z^T x - x^T P x - q^T x - r` is

          :math:`z - 2 P x - q`

        hence, the argsup can be obtained when x makes the gradient zero
         (when P is positive definite), which is

          :math:`(1/2) P^{-1} (z-q)`
        """
        assert self.is_strictly_convex
        assert self.num_inputs is None or z_array_2d.shape[1] == self.num_inputs

        x_array_2d_list: list[ndarray] = list()

        assert self.quad_3d is not None
        for idx3 in range(self.quad_3d.shape[2]):
            p_array_2d: ndarray = self.quad_3d[:, :, idx3]
            q_array_1d: ndarray = self.slope_2d[:, idx3]

            assert z_array_2d.shape[1] == q_array_1d.size

            x_array_2d: ndarray = la.solve(p_array_2d, (z_array_2d - q_array_1d).T).T / 2.0

            assert x_array_2d.shape == z_array_2d.shape

            x_array_2d_list.append(x_array_2d)

        x_array_3d: ndarray = stack(x_array_2d_list, axis=2)

        assert x_array_3d.shape == (
            z_array_2d.shape[0],
            self.num_inputs,
            self.num_outputs,
        )

        return x_array_3d

    def _get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        logger.debug(x_array_2d.shape)
        logger.debug(self.slope_2d.shape)
        logger.debug(self.intercept_1d.shape)
        y_array_2d = x_array_2d.dot(self.slope_2d) + self.intercept_1d

        if self.quad_3d is not None:
            for idx in range(self.quad_3d.shape[2]):
                y_array_2d[:, idx] += (x_array_2d.dot(self.quad_3d[:, :, idx]) * x_array_2d).sum(
                    axis=1
                )

        return y_array_2d

    def _jacobian(self, x_array_2d: ndarray) -> ndarray:

        jac: np.ndarray = (
            self.slope_2d[:, :, None].transpose([2, 1, 0]).repeat(x_array_2d.shape[0], axis=0)
        )

        if self.quad_3d is not None:
            jac += np.array(
                [
                    2.0 * np.dot(x_array_2d, self.quad_3d[:, :, idx])
                    for idx in range(self.quad_3d.shape[2])
                ]
            ).transpose([1, 0, 2])

        return jac

    def _hessian(self, x_array_2d: np.ndarray) -> np.ndarray:
        num_data: int = x_array_2d.shape[0]
        if self.quad_3d is None:
            return np.zeros((num_data, self.num_outputs, self.num_inputs, self.num_inputs))

        return self.quad_3d.transpose([2, 0, 1])[np.newaxis].repeat(num_data, axis=0)
