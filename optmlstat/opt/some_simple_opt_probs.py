"""
some simple optimization problems
"""

import numpy as np
from numpy.random import randn

from optmlstat.basic_modules.class_base import OMSClassBase
from optmlstat.opt.opt_prob import OptProb
from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.functions.basic_functions.affine_function import AffineFunction


class SomeSimpleOptProbs(OMSClassBase):
    @staticmethod
    def get_simple_linear_program_two_vars(num_points: int, /) -> tuple[OptProb, np.ndarray]:
        """
        :param num_points: # randomly generated feasible points
        :return: optimization problem

        simple LP
        min. x^2 + y^2
        s.t. x + y <= 1

        opt_sol = (1/2, 1/2)
        opt_val = 1/2

        second output is randomly generated feasible points
        """

        rand_array_1d: np.ndarray = randn(num_points)

        return (
            OptProb(
                QuadraticFunction(np.eye(2)[:, :, np.newaxis], np.zeros((2, 1)), np.zeros(1)),
                AffineFunction(np.ones((2, 1)), -np.ones(1)),
                None,
            ),
            np.vstack([rand_array_1d, 1.0 - rand_array_1d]).T,
        )
