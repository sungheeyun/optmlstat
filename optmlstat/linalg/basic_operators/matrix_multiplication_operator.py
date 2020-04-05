from optmlstat.linalg.basic_operators.linear_vector_operator_base import LinearVectorOperatorBase

from numpy import ndarray


class MatrixMultiplicationOperator(LinearVectorOperatorBase):
    """
    Defines a linear operator by a 2-d array.
    This defines the following *matrix multiplication*.

    f(x) = A x

    where x is a column vector.

    Since this class takes 1-d array, this may create confusion,
    but keep in mind that the 1-d array given is interpreted as a column vector,
    and the 2-d array taken as constructor argument is interpreted as matrix
    as shown in the above equation.
    """

    def __init__(self, a_array_2d: ndarray) -> None:
        super(MatrixMultiplicationOperator, self).__init__(a_array_2d.shape[1], a_array_2d.shape[0])
        self.a_array_2d_T: ndarray = a_array_2d.T

    def transform(self, input_array_1d: ndarray) -> ndarray:
        return input_array_1d.dot(self.a_array_2d_T)
