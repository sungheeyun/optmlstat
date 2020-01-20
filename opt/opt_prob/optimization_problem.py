from typing import Optional, Union

from basic_modueles.class_base import OptMLStatClassBase
from functions.function_base import FunctionBase


class OptimizationProblem(OptMLStatClassBase):
    def __init__(
        self,
        obj_fcn: Optional[FunctionBase] = None,
        eq_cnst: Optional[FunctionBase] = None,
        ineq_cnst: Optional[FunctionBase] = None,
    ) -> None:
        self.obj_fcn: Optional[FunctionBase] = obj_fcn
        self.eq_cnst: Optional[FunctionBase] = eq_cnst
        self.ineq_const: Optional[FunctionBase] = ineq_cnst

    def to_json_data(self) -> Union[int, float, str, dict, list]:
        return dict(
            class_category="OptimizationProblem",
            obj_fcn=None if self.obj_fcn is None else self.obj_fcn.to_json_data(),
            eq_cnst=None if self.eq_cnst is None else self.eq_cnst.to_json_data(),
            ineq_cnst=None if self.ineq_const is None else self.ineq_const.to_json_data(),
        )
