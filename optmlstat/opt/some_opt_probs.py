"""
some simple optimization problems
"""

import numpy as np
from numpy.random import randn
from scipy import linalg

from optmlstat.basic_modules.class_base import OMSClassBase
from optmlstat.opt.opt_prob import OptProb
from optmlstat.functions.function_base import FunctionBase
from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.functions.basic_functions.log_sum_exp import LogSumExp
from optmlstat.functions.basic_functions.affine_function import AffineFunction
from optmlstat.functions.basic_functions.standard_affine_function import StandardAffineFunction
from optmlstat.linalg.utils import get_random_pos_def_array


class SomeSimpleOptProbs(OMSClassBase):
    @staticmethod
    def simple_linear_program_two_vars(num_points: int, /) -> tuple[OptProb, np.ndarray]:
        """
        :param num_points: # randomly generated feasible points
        :return: optimization problem

        simple LP
        min. x^2 + y^2
        s.t. x + y = 1

        opt_sol = (1/2, 1/2)
        opt_val = 1/2

        second output is randomly generated feasible points
        """

        rand_1d: np.ndarray = randn(num_points)

        return (
            OptProb(
                QuadraticFunction(np.eye(2)[:, :, np.newaxis], np.zeros((2, 1)), np.zeros(1)),
                AffineFunction(np.ones((2, 1)), -np.ones(1)),
                None,
            ),
            np.vstack([rand_1d, 1.0 - rand_1d]).T,
        )

    @classmethod
    def random_eq_const_cvx_quad_prob(
        cls, num_vars: int, num_eq_cnsts: int, num_points: int, /
    ) -> tuple[OptProb, np.ndarray]:
        """
        min. \|x\|_2^2  # noqa:W605
        s.t. Ax = b
        """
        return cls._random_eq_const_prob(
            num_vars,
            num_eq_cnsts,
            num_points,
            QuadraticFunction(
                get_random_pos_def_array(num_vars)[:, :, np.newaxis], randn(num_vars, 1), randn(1)
            ),
        )

    @classmethod
    def random_eq_const_lse_prob(
        cls, num_vars: int, num_eq_cnsts: int, num_points: int, /
    ) -> tuple[OptProb, np.ndarray]:
        """
        min. sum exp (Cx+d)
        s.t. Ax = b
        :return:
        - optimization problem
        - feasible points
        """
        return cls._random_eq_const_prob(
            num_vars,
            num_eq_cnsts,
            num_points,
            LogSumExp([1e-0 * randn(num_vars * 3, num_vars)], 1e-0 * randn(1, num_vars * 3)),
        )

    @staticmethod
    def _random_eq_const_prob(
        num_vars: int, num_eq_cnsts: int, num_points: int, objfcn: FunctionBase
    ) -> tuple[OptProb, np.ndarray]:
        """
        :return:
        - optimization problem
        - feasible points
        """

        assert num_eq_cnsts < num_vars, (num_eq_cnsts, num_vars)

        _a_2d: np.ndarray = randn(num_eq_cnsts, num_vars)
        _b_1d: np.ndarray = randn(num_eq_cnsts)

        eq_cnst: StandardAffineFunction = StandardAffineFunction(_a_2d, -_b_1d)

        orth_2d: np.ndarray = linalg.qr(_a_2d.T)[0]
        null_2d: np.ndarray = orth_2d[:, num_eq_cnsts:]

        feas_sol: np.ndarray = linalg.lstsq(_a_2d, _b_1d)[0]
        feas_pnts: np.ndarray = (
            np.dot(null_2d, 2.0 * randn(null_2d.shape[1], num_points)).T + feas_sol
        )
        assert np.allclose(eq_cnst.get_y_values_2d(feas_pnts), 0.0), eq_cnst.get_y_values_2d(
            feas_pnts
        )

        return OptProb(objfcn, eq_cnst, None), feas_pnts
