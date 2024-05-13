from typing import Any

from numpy import ndarray

from optmlstat.basic_modules.class_base import OMSClassBase
from optmlstat.functions.function_base import FunctionBase
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

        self._num_eq_cnst: int
        if self.eq_cnst_fcn is None:
            self._num_eq_cnst = 0
        else:
            assert self.eq_cnst_fcn.num_outputs is not None
            self._num_eq_cnst = self.eq_cnst_fcn.num_outputs

        self._num_ineq_cnst: int
        if self.ineq_cnst_fcn is None:
            self._num_ineq_cnst = 0
        else:
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
    def domain_dim(self) -> int:
        return self._domain_dim

    @property
    def num_eq_cnst(self) -> int:
        return self._num_eq_cnst

    @property
    def num_ineq_cnst(self) -> int:
        return self._num_ineq_cnst

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
            None if self.obj_fcn is None else self.obj_fcn.jacobian(x_array_2d)
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
