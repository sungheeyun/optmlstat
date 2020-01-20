from typing import Optional

from functions.function_base import FunctionBase


class OptimizationProblem:
    def __init__(
        self, obj_fcn: Optional[FunctionBase], eq_const: Optional[FunctionBase], ineq_const: Optional[FunctionBase]
    ) -> None:
        self.obj_fcn: Optional[FunctionBase] = obj_fcn
        self.eq_const: Optional[FunctionBase] = eq_const
        self.ineq_const: Optional[FunctionBase] = ineq_const
