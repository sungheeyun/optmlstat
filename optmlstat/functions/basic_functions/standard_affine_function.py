"""
affine functions where A is m-by-n matrix where n is # vars
"""

from numpy import ndarray

from optmlstat.functions.basic_functions.affine_function import AffineFunction


class StandardAffineFunction(AffineFunction):
    """
    affine function with standard definition of coefficient matrix
    """

    def __init__(self, slope_array_2d: ndarray, intercept_array_1d: ndarray) -> None:
        super().__init__(slope_array_2d.T, intercept_array_1d)

    # @property
    # def slope_array_2d(self) -> ndarray:
    #     return self._slope_array_2d.T
