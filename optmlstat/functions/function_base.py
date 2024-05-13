"""
base class for all functions in optmlstat package
"""

from __future__ import annotations
from abc import abstractmethod

import numpy as np

from optmlstat.basic_modules.class_base import OMSClassBase
from optmlstat.functions.fcn_decorators import (
    fcn_evaluator,
    differentiable_fcn_evaluator,
    twice_differentiable_fcn_evaluator,
)


class FunctionBase(OMSClassBase):
    """
    The base class for function classes.

    A function represented by a subclass of this class is a function whose domain is n-dimensional
    (Euclidean) vector space
     and the range is m-dimensional vector space in general where m and n are positive integers.
    """

    @property
    @abstractmethod
    def num_inputs(self) -> int:
        """
        The number of inputs, i.e., the dimension of the domain of the function.
        """
        pass

    @property
    @abstractmethod
    def num_outputs(self) -> int:
        """
        The number of outputs, i.e., the dimension of the range of the function.
        """
        pass

    @property
    @abstractmethod
    def is_affine(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_strictly_convex(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_convex(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_strictly_concave(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_concave(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_differentiable(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_twice_differentiable(self) -> bool:
        pass

    # TODO (2) implemented the below method for all subclasses of FunctionBase

    def get_shape(self) -> tuple[int, int]:
        return self.num_inputs, self.num_outputs

    def check_x_array_dimension(self, x_array_2d: np.ndarray) -> None:
        if self.num_inputs is not None:
            assert x_array_2d.shape[1] == self.num_inputs

    def check_y_array_dimension(self, y_array_2d: np.ndarray) -> None:
        if self.num_outputs is not None:
            assert y_array_2d.shape[1] == self.num_outputs

    def get_y_values_2d_from_x_values_1d(self, x_array_1d: np.ndarray) -> np.ndarray:
        x_array_2d: np.ndarray = np.array([x_array_1d]).T
        return self.get_y_values_2d(x_array_2d)

    def eval(self, x_array_2d: np.ndarray) -> np.ndarray:
        return self.get_y_values_2d(x_array_2d)

    @differentiable_fcn_evaluator
    @fcn_evaluator
    def jacobian(self, x_array_2d: np.ndarray) -> np.ndarray:
        """
        output array index - [data, functions, x]
        """
        return self._jacobian(x_array_2d)
        pass

    @abstractmethod
    def _jacobian(self, x_array_2d: np.ndarray) -> np.ndarray:
        pass

    @twice_differentiable_fcn_evaluator
    @fcn_evaluator
    def hessian(self, x_array_2d: np.ndarray) -> np.ndarray:
        """
        output array index - [data, functions, x, x]
        """
        return self._hessian(x_array_2d)

    @abstractmethod
    def _hessian(self, x_array_2d: np.ndarray) -> np.ndarray:
        pass

    @fcn_evaluator
    def get_y_values_2d(self, x_array_2d: np.ndarray) -> np.ndarray:
        """
        Returns y values for given x values. Each row of x_array_2d represents
        each x vector (n-dimensional vector) and each row of y_array_2d
        represents the corresponding y value (m-dimensional vector).
        Unlike generally accepted linear algebra standard,
        we use row-vector representation for each data point
        to conform to Machine Learning convention where each row represents data
        and each column represent the feature.

        More precisely, the function value for x_array_2d[i, :] is stored
        in y_array_2d[i, :].

        Parameters
        ----------
        x_array_2d: N-by-n np.array representing x.

        Returns
        -------
        y_array_2d: N-by-m np.array representing y.
        """
        return self._get_y_values_2d(x_array_2d)

    @abstractmethod
    def _get_y_values_2d(self, x_array_2d: np.ndarray) -> np.ndarray:
        pass

    # TODO (2) define decorator
    #  to check whether a function is convex for conjugate and conjugate_arg

    @property
    @abstractmethod
    def conjugate(self) -> FunctionBase:
        """
        Returns the conjugate of this function:

          :math:`f^\ast (z) = sup_x ( z^T x - f(x) )`

        Since is a supremum of a linear function (in x), this function is always convex.
        It also has the following property (by definition): for all x and y, it satisfies

          :math:`f(x) + f^\ast(z) >= x^T z`.

        Note that the domain of this function is also n-dimensional vector space,
        but it represents a different vector space than the domain of :math:`f`.
        """
        pass

    @abstractmethod
    def conjugate_arg(self, z_array_2d: np.ndarray) -> np.ndarray:
        """
        Returns the x values which attain the conjugate function for :math:`z`, i.e.,

          :math:`argsup_x ( z^T x - f(x) )`

        Note that for each :math:`z`, the output is a n-dimensional vector
        (not a scalar as in the case of normal
        function evaluation by get_y_values_2d, the return value of this function
        is 3-dimensional np.array.
        (Note here 'dimensional' is used for different meanings.
        You should understand it correctly in the context.)

        More precisely, the arg_sup value of z_array_2d[i, :]
        for the jth output (function) is stored in x_array_3d[i, :, j].

        Parameters
        ----------
        z_array_2d:
          N-by-n np.array representing :math:`z`.

        Returns
        -------
        x_array_3d:
          N-by-m-by-n np.array representing argsup :math:`(z^T x - f(x))`.
        """
        pass
