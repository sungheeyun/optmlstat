"""
class for optimization problems
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import ndarray
from scipy import linalg

from optmlstat.basic_modules.class_base import OMSClassBase
from optmlstat.functions.basic_functions.affine_function import AffineFunction
from optmlstat.functions.basic_functions.quadratic_function import QuadraticFunction
from optmlstat.functions.exceptions import ValueUnknownException
from optmlstat.functions.basic_functions.constant_function import ConstantFunction
from optmlstat.functions.function_base import FunctionBase
from optmlstat.functions.special_functions.empty_function import EmptyFunction
from optmlstat.linalg.utils import block_array
from optmlstat.opt.opt_prob_eval import OptProbEval


class OptProb(OMSClassBase):
    """
    A general mathematical optimization problem
    """

    def __init__(
        self,
        obj_fcn: FunctionBase | None = None,
        eq_cnst: FunctionBase | None = None,
        ineq_cnst: FunctionBase | None = None,
        /,
    ) -> None:
        """
        The optimization problem is

        minimize obj_fcn
        subject to eq_cnst_fcn = 0
                   ineq_cnst_fcn <= 0

        Parameters
        ----------
        obj_fcn:
         objective function
        eq_cnst:
         equality constraint function
        ineq_cnst:
         inequality constraint function
        """

        assert not (obj_fcn is None and eq_cnst is None and ineq_cnst is None)

        self.obj_fcn: FunctionBase | None = obj_fcn
        self.eq_cnst_fcn: FunctionBase | None = eq_cnst
        self.ineq_cnst_fcn: FunctionBase | None = ineq_cnst

        self._num_objs: int = 0
        if self.obj_fcn is not None:
            assert self.obj_fcn.num_inputs is not None
            self._num_objs = self.obj_fcn.num_outputs

        self._num_eq_cnst: int = 0
        if self.eq_cnst_fcn is not None:
            assert self.eq_cnst_fcn.num_outputs is not None
            self._num_eq_cnst = self.eq_cnst_fcn.num_outputs

        self._num_ineq_cnst: int = 0
        if self.ineq_cnst_fcn is not None:
            assert self.ineq_cnst_fcn.num_outputs is not None
            self._num_ineq_cnst = self.ineq_cnst_fcn.num_outputs

        domain_dim: int | None = None
        if self.obj_fcn is not None:
            if domain_dim is None:
                domain_dim = self.obj_fcn.num_inputs
            else:
                assert domain_dim == self.obj_fcn.num_inputs

        if self.eq_cnst_fcn is not None:
            if domain_dim is None:
                domain_dim = self.eq_cnst_fcn.num_inputs
            else:
                assert domain_dim == self.eq_cnst_fcn.num_inputs

        if self.ineq_cnst_fcn is not None:
            if domain_dim is None:
                domain_dim = self.ineq_cnst_fcn.num_inputs
            else:
                assert domain_dim == self.ineq_cnst_fcn.num_inputs

        assert isinstance(domain_dim, int)
        self._domain_dim: int = domain_dim

        self._is_convex: bool = True
        if self.obj_fcn is not None and not self.obj_fcn.is_convex:
            self._is_convex = False

        if self.eq_cnst_fcn is not None and not self.eq_cnst_fcn.is_affine:
            self._is_convex = False

        if self.ineq_cnst_fcn is not None and not self.ineq_cnst_fcn.is_convex:
            self._is_convex = False

    @property
    def dual_problem(self) -> OptProb:
        num_dual_variables: int = self.num_ineq_cnst + self.num_eq_cnst

        if num_dual_variables == 0:
            objfcn: FunctionBase = EmptyFunction(num_dual_variables, 1)
            try:
                objfcn = ConstantFunction(-self.optimum_value, num_dual_variables)
            except ValueUnknownException:
                pass

            return OptProb(
                objfcn,
                EmptyFunction(num_dual_variables, 0),
                AffineFunction(
                    block_array(
                        [
                            [
                                -np.eye(self.num_ineq_cnst),
                                np.zeros((self.num_ineq_cnst, self.num_eq_cnst)),
                            ]
                        ]
                    ).T,
                    np.zeros(self.num_ineq_cnst),
                ),
            )

        if self.num_ineq_cnst == 0 and isinstance(self.eq_cnst_fcn, AffineFunction):
            if isinstance(self.obj_fcn, QuadraticFunction):
                _a_2d: np.ndarray = self.eq_cnst_fcn.slope_2d.T
                _b_1d: np.ndarray = -self.eq_cnst_fcn.intercept_1d
                assert isinstance(self.obj_fcn.quad_3d, np.ndarray), self.obj_fcn.quad_3d.__class__
                dual_quad_3d: np.ndarray = np.array(
                    [
                        -np.dot(_a_2d, linalg.solve(quad_array_2d, _a_2d.T, assume_a="sym")) / 4.0
                        for quad_array_2d in self.obj_fcn.quad_3d.transpose((2, 0, 1))
                    ]
                ).transpose((1, 2, 0))

                dual_slope_2d: np.ndarray = np.array(
                    [
                        -_b_1d
                        - np.dot(_a_2d, linalg.solve(self.obj_fcn.quad_3d[:, :, idx], slope_1d))
                        / 2.0
                        for idx, slope_1d in enumerate(self.obj_fcn.slope_2d.T)
                    ]
                ).T

                dual_intercept_1d: np.ndarray = np.array(
                    [
                        -np.dot(
                            self.obj_fcn.slope_2d[:, idx],
                            linalg.solve(
                                self.obj_fcn.quad_3d[:, :, idx],
                                self.obj_fcn.slope_2d[:, idx],
                                assume_a="sym",
                            ),
                        )
                        / 4.0
                        + float(intercept)
                        for idx, intercept in enumerate(self.obj_fcn.intercept_1d)
                    ]
                )
                return OptProb(QuadraticFunction(-dual_quad_3d, -dual_slope_2d, -dual_intercept_1d))

        return OptProb(
            EmptyFunction(num_dual_variables, 1),
            EmptyFunction(num_dual_variables, 0),
            AffineFunction(
                block_array(
                    [
                        [
                            -np.eye(self.num_ineq_cnst),
                            np.zeros((self.num_ineq_cnst, self.num_eq_cnst)),
                        ]
                    ]
                ).T,
                np.zeros(self.num_ineq_cnst),
            ),
        )

    @property
    def dim_domain(self) -> int:
        return self._domain_dim

    @property
    def num_objs(self) -> int:
        return self._num_objs

    @property
    def num_eq_cnst(self) -> int:
        return self._num_eq_cnst

    @property
    def num_ineq_cnst(self) -> int:
        return self._num_ineq_cnst

    @property
    def optimum_value(self) -> np.ndarray:
        """
        1-d vector in output space
        here "optimum" means *minimum* vector, not *minimal* vector
        in the sense of multi-objective optimization
        """
        return self.optimum_x_lambda_nu_val[3]

    @property
    def optimum_point(self) -> np.ndarray:
        """
        1-d vector in input space
        here "optimum" means *minimum* vector, not *minimal* vector
        in the sense of multi-objective optimization
        """
        return self.optimum_x_lambda_nu_val[0]

    @property
    def optimum_lambda(self) -> np.ndarray:
        return self.optimum_x_lambda_nu_val[1]

    @property
    def optimum_nu(self) -> np.ndarray:
        return self.optimum_x_lambda_nu_val[2]

    @property
    def optimum_x_lambda_nu_val(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.num_objs == 1:
            if self.num_eq_cnst == 0 and self.num_ineq_cnst == 0:
                assert self.obj_fcn is not None
                return (
                    self.obj_fcn.minimum_point,
                    np.ndarray(0),
                    np.ndarray(0),
                    self.obj_fcn.minimum_value,
                )

            if self.num_ineq_cnst == 0 and isinstance(self.eq_cnst_fcn, AffineFunction):
                # linearly eq cnst opt
                if isinstance(self.obj_fcn, QuadraticFunction) and self.obj_fcn.is_strictly_convex:
                    assert self.obj_fcn.quad_3d is not None
                    _p_2d: np.ndarray = self.obj_fcn.quad_3d[:, :, 0]
                    _q_1d: np.ndarray = self.obj_fcn.slope_2d[:, 0]
                    # _r_0d: float = self.obj_fcn.intercept_1d[0]
                    _a_2d: np.ndarray = self.eq_cnst_fcn.slope_2d.T
                    _b_1d: np.ndarray = -self.eq_cnst_fcn.intercept_1d
                    x_nu_1d: np.ndarray = linalg.solve(
                        block_array(
                            [
                                [
                                    2.0 * _p_2d,
                                    _a_2d.T,
                                ],
                                [_a_2d, 0.0],
                            ]
                        ),
                        block_array([-_q_1d, _b_1d]),
                        assume_a="sym",
                    )
                    opt_x: np.ndarray = x_nu_1d[: self.dim_domain]
                    opt_nu: np.ndarray = x_nu_1d[self.dim_domain :]  # noqa:E203
                    return (
                        opt_x,
                        np.ndarray(0),
                        opt_nu,
                        self.obj_fcn.get_y_values_2d(opt_x[np.newaxis, :])[0],
                    )

        raise ValueUnknownException()

    @property
    def optimal_value(self) -> np.ndarray | float:
        """
        1-d vector in output space
        here "optimal means" *minimal* vector, not *minimum* vector
        in the sense of multi-objective optimization
        """
        if self.num_objs == 1 and self.num_eq_cnst == 0 and self.num_ineq_cnst == 0:
            return self.optimum_value
        raise NotImplementedError()

    @property
    def optimal_point(self) -> np.ndarray:
        """
        1-d vector in output space
        here "optimal means" *minimal* vector, not *minimum* vector
        in the sense of multi-objective optimization
        """
        if self.num_objs == 1 and self.num_eq_cnst == 0 and self.num_ineq_cnst == 0:
            return self.optimum_point
        raise NotImplementedError()

    @property
    def is_convex(self) -> bool:
        return self._is_convex

    def to_json_data(self) -> dict[str, Any]:
        return dict(
            class_category="OptimizationProblem",
            obj_fcn=None if self.obj_fcn is None else self.obj_fcn.to_json_data(),
            eq_cnst=(None if self.eq_cnst_fcn is None else self.eq_cnst_fcn.to_json_data()),
            ineq_cnst=(None if self.ineq_cnst_fcn is None else self.ineq_cnst_fcn.to_json_data()),
        )

    def evaluate(self, x_array_2d: ndarray) -> OptProbEval:

        obj_fcn_jac_3d: ndarray | None = (
            None
            if self.obj_fcn is None
            else (self.obj_fcn.jacobian(x_array_2d) if self.obj_fcn.is_differentiable else None)
        )

        return OptProbEval(
            opt_prob=self,
            x_array_2d=x_array_2d.copy(),
            obj_fcn_array_2d=(
                None if self.obj_fcn is None else self.obj_fcn.get_y_values_2d(x_array_2d)
            ),
            obj_fcn_jac_3d=obj_fcn_jac_3d,
            eq_cnst_array_2d=(
                None if self.eq_cnst_fcn is None else self.eq_cnst_fcn.get_y_values_2d(x_array_2d)
            ),
            ineq_cnst_array_2d=(
                None
                if self.ineq_cnst_fcn is None
                else self.ineq_cnst_fcn.get_y_values_2d(x_array_2d)
            ),
        )
