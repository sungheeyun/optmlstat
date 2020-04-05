from typing import Any, Optional, Dict
from dataclasses import dataclass
import json

from numpy import ndarray

from optmlstat.formatting import convert_data_for_json


@dataclass(frozen=True)
class OptimizationProblemEvaluation:
    opt_prob: Any
    x_array_2d: ndarray
    obj_fcn_array_2d: Optional[ndarray] = None
    ineq_cnst_array_2d: Optional[ndarray] = None
    eq_cnst_array_2d: Optional[ndarray] = None

    def to_json_data(self) -> Dict[str, Any]:
        return dict(
            opt_prob=self.opt_prob,
            x_array_2d=self.x_array_2d,
            obj_fcn_array_2d=self.obj_fcn_array_2d,
            ineq_cnst_array_2d=self.ineq_cnst_array_2d,
            eq_cnst_array_2d=self.ineq_cnst_array_2d,
        )

    def __repr__(self) -> str:
        return json.dumps(self.to_json_data(), indent=2, default=convert_data_for_json)
