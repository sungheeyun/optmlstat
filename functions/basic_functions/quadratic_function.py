from __future__ import annotations
from typing import Optional, List, Tuple
from logging import Logger, getLogger

from numpy import ndarray, vstack, array
from numpy.linalg import eig, inv, solve

from functions.function_base import FunctionBase


logger: Logger = getLogger()


class QuadraticFunction(FunctionBase):
    """
    Quadratic function. The ith function is

      f_i(x) = x^T P x + q^T x + r

    where x is n-dimensional real column-vector, P is an real n-by-n matrix, q is an real n-dimensional
    column vector, and r is a real scalar. i goes from 1 to m.

    Note that when data are actually interchanged, e.g., though method, QuadraticFunction.get_y_array_2d,
    x is represented by row vectors (not column vectors).
    """

    @staticmethod
    def test_convexity(quad_array_3d: ndarray) -> Tuple[bool, bool]:
        is_strictly_convex: bool = True
        is_convex: bool = True
        for idx3 in range(quad_array_3d.shape[2]):
            symmetric_array: ndarray = quad_array_3d[:, :, idx3] + quad_array_3d[:, :, idx3].T
            eigen_value_array: ndarray = eig(symmetric_array)[0]
            if (eigen_value_array <= 0.0).any():
                is_strictly_convex = False

            if (eigen_value_array < 0.0).any():
                is_convex = False

        return is_convex, is_strictly_convex

    def __init__(self, quad_array_3d: Optional[ndarray], slope_array_2d: ndarray, intercept_array_1d: ndarray) -> None:
        """
        If n is the number of inputs and m is that of outputs, quad_array_3d[:, :, i], slope_array_2d[:, i] and
        intercept_array_1d[i] represents P, q, and r respectively in the following equation:

          f_i(x) = x^T P x + q^T x + r

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
        assert slope_array_2d.shape[1] == intercept_array_1d.size, (slope_array_2d.shape, intercept_array_1d.shape)
        assert quad_array_3d is None or quad_array_3d.shape[2] == intercept_array_1d.size, (
            slope_array_2d.shape,
            intercept_array_1d.shape,
        )

        self.intercept_array_1d: ndarray = intercept_array_1d.copy()
        self.slope_array_2d: ndarray = slope_array_2d.copy()
        self.quad_array_3d: ndarray = None if quad_array_3d is None else quad_array_3d.copy()

        self._is_affine: bool = (self.quad_array_3d == 0.0).all()
        self._is_convex: bool
        self._is_strictly_convex: bool
        self._is_concave: bool
        self._is_strictly_concave: bool

        self._is_convex, self._is_strictly_convex = QuadraticFunction.test_convexity(self.quad_array_3d)
        self._is_concave, self._is_strictly_concave = QuadraticFunction.test_convexity(- self.quad_array_3d)

    @property
    def num_inputs(self) -> Optional[int]:
        return self.slope_array_2d.shape[0]

    @property
    def num_outputs(self) -> Optional[int]:
        return self.slope_array_2d.shape[1]

    @property
    def is_affine(self) -> Optional[bool]:
        return self._is_affine

    @property
    def is_convex(self) -> Optional[bool]:
        return self._is_convex

    @property
    def is_strictly_convex(self) -> Optional[bool]:
        return self._is_strictly_convex

    @property
    def is_concave(self) -> Optional[bool]:
        return self._is_concave

    @property
    def is_strictly_concave(self) -> Optional[bool]:
        return self._is_strictly_concave

    @property
    def conjugate(self) -> QuadraticFunction:
        """
        If the (quadratic) function is not convex, this value can be infinity. (not always though)
        If the function is convex, but not strictly convex, the conjugate is bounded,
         but the calculation is a bit complicated.
        For now, we implement this function only when the function is strictly convex.

        In this case, when f(x) = x^T P x + q^T x + r,
        the conjugate is (y^T P^{-1} y - 2 q^T P^{-1} y + q^T P^{-1} q)/4 - r.
        """
        assert self.is_strictly_convex

        conjugate_quad_array_3d: ndarray = ndarray(shape=self.quad_array_3d.shape, dtype=float)
        conjugate_slope_array_1d_list: List[ndarray] = list()
        conjugate_intercept_list: List[float] = list()

        for idx in range(self.quad_array_3d.shape[2]):
            p_array_2d: ndarray = self.quad_array_3d[:, :, idx]
            conjugate_quad_array_3d[:, :, idx] = 0.25 * inv(p_array_2d)

            q_array_1d: ndarray = self.slope_array_2d[:, idx]
            p_inv_q_array_1d: ndarray = solve(p_array_2d, q_array_1d)
            conjugate_slope_array_1d_list.append(- 0.5 * p_inv_q_array_1d)

            conjugate_intercept_list.append(0.25 * q_array_1d.dot(p_inv_q_array_1d))

        conjugate_slope_array_2d: ndarray = vstack(conjugate_slope_array_1d_list).T
        conjugate_intercept_array_1d: ndarray = array(conjugate_intercept_list, float) - self.intercept_array_1d

        logger.debug(f"conjugate_slope_array_1d_list: {conjugate_slope_array_1d_list}")
        logger.debug(f"conjugate_slope_array_2d.shape: {conjugate_slope_array_2d.shape}")

        return QuadraticFunction(conjugate_quad_array_3d, conjugate_slope_array_2d, conjugate_intercept_array_1d)

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        logger.debug(x_array_2d.shape)
        logger.debug(self.slope_array_2d.shape)
        logger.debug(self.intercept_array_1d.shape)
        y_array_2d = x_array_2d.dot(self.slope_array_2d) + self.intercept_array_1d

        if self.quad_array_3d is not None:
            for idx in range(self.quad_array_3d.shape[2]):
                y_array_2d[:, idx] += (x_array_2d.dot(self.quad_array_3d[:, :, idx]) * x_array_2d).sum(axis=1)

        return y_array_2d
