from typing import Optional, Union

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
        subject to eq_cnst = 0
                   ineq_cnst <= 0

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
        self.eq_cnst: Optional[FunctionBase] = eq_cnst
        self.ineq_cnst: Optional[FunctionBase] = ineq_cnst

        domain_dim: Optional[int] = None
        if self.obj_fcn is not None:
            if domain_dim is None:
                domain_dim = self.obj_fcn.num_inputs
            else:
                assert domain_dim == self.obj_fcn.num_inputs

        if self.eq_cnst is not None:
            if domain_dim is None:
                domain_dim = self.eq_cnst.num_inputs
            else:
                assert domain_dim == self.eq_cnst.num_inputs

        if self.ineq_cnst is not None:
            if domain_dim is None:
                domain_dim = self.ineq_cnst.num_inputs
            else:
                assert domain_dim == self.ineq_cnst.num_inputs

        assert isinstance(domain_dim, int)
        self.domain_dim: int = domain_dim

        self.is_convex: bool = True
        if self.obj_fcn is not None and not self.obj_fcn.is_convex:
            self.is_convex = False

        if self.eq_cnst is not None and not self.eq_cnst.is_affine:
            self.is_convex = False

        if self.ineq_cnst is not None and not self.ineq_cnst.is_convex:
            self.is_convex = False

    def to_json_data(self) -> Union[int, float, str, dict, list]:
        return dict(
            class_category="OptimizationProblem",
            obj_fcn=None if self.obj_fcn is None else self.obj_fcn.to_json_data(),
            eq_cnst=None if self.eq_cnst is None else self.eq_cnst.to_json_data(),
            ineq_cnst=None if self.ineq_cnst is None else self.ineq_cnst.to_json_data(),
        )
