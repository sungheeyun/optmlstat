from typing import Optional, Any, Dict

from numpy import ndarray

from basic_modueles.class_base import OptMLStatClassBase
from functions.function_base import FunctionBase


class OptimizationProblem(OptMLStatClassBase):
    """
    A general mathematical optimization problem
    """

    def __init__(
        self,
        obj_fcn: Optional[FunctionBase] = None,
        eq_cnst: Optional[FunctionBase] = None,
        ineq_cnst: Optional[FunctionBase] = None,
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

        self.obj_fcn: Optional[FunctionBase] = obj_fcn
        self.eq_cnst_fcn: Optional[FunctionBase] = eq_cnst
        self.ineq_cnst_fcn: Optional[FunctionBase] = ineq_cnst
        self._num_eq_cnst: int = 0 if self.eq_cnst_fcn is None else self.eq_cnst_fcn.num_outputs
        self._num_ineq_cnst: int = 0 if self.ineq_cnst_fcn is None else self.ineq_cnst_fcn.num_outputs

        domain_dim: Optional[int] = None
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

        self.is_convex: bool = True
        if self.obj_fcn is not None and not self.obj_fcn.is_convex:
            self.is_convex = False

        if self.eq_cnst_fcn is not None and not self.eq_cnst_fcn.is_affine:
            self.is_convex = False

        if self.ineq_cnst_fcn is not None and not self.ineq_cnst_fcn.is_convex:
            self.is_convex = False

    @property
    def domain_dim(self) -> int:
        return self._domain_dim

    @property
    def num_eq_cnst(self) -> int:
        return self._num_eq_cnst

    @property
    def num_ineq_cnst(self):
        return self._num_ineq_cnst

    def to_json_data(self, x_array_2d: Optional[ndarray] = None) -> Dict[str, Any]:
        if x_array_2d is None:
            return dict(
                class_category="OptimizationProblem",
                obj_fcn=None if self.obj_fcn is None else self.obj_fcn.to_json_data(),
                eq_cnst=None if self.eq_cnst_fcn is None else self.eq_cnst_fcn.to_json_data(),
                ineq_cnst=None if self.ineq_cnst_fcn is None else self.ineq_cnst_fcn.to_json_data(),
            )
        else:
            return dict(
                class_category="OptimizationProblemEvaluated",
                obj_fcn=None if self.obj_fcn is None else self.obj_fcn.get_y_values_2d(x_array_2d),
                eq_cnst=None if self.eq_cnst_fcn is None else self.eq_cnst_fcn.get_y_values_2d(x_array_2d),
                ineq_cnst=None if self.ineq_cnst_fcn is None else self.ineq_cnst_fcn.get_y_values_2d(x_array_2d),
            )
