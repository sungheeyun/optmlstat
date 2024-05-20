"""
optimization problem evaluation results
"""

from typing import Any
from dataclasses import dataclass
import json

import numpy as np
from numpy import ndarray

from optmlstat.formatting import convert_data_for_json


def ndarray_to_list(a: np.ndarray | float) -> list | float:
    if isinstance(a, np.ndarray):
        return [ndarray_to_list(x) for x in a]
    else:
        return float(a)


@dataclass(frozen=True)
class OptProbEval:
    opt_prob: Any
    x_2d: ndarray
    obj_fcn_2d: ndarray | None = None
    obj_fcn_jac_3d: ndarray | None = None
    obj_fcn_hess_4d: ndarray | None = None
    ineq_cnst_2d: ndarray | None = None
    ineq_cnst_jac_3d: ndarray | None = None
    ineq_cnst_hess_4d: ndarray | None = None
    eq_cnst_2d: ndarray | None = None
    eq_cnst_jac_3d: ndarray | None = None
    eq_cnst_hess_4d: ndarray | None = None

    def to_json_data(self) -> dict[str, Any]:
        return dict(
            pt_prob=self.opt_prob,
            x_2d=self.x_2d,
            obj_fcn_2d=self.obj_fcn_2d,
            obj_fcn_jac_3d=self.obj_fcn_jac_3d,
            obj_fcn_hess_4d=self.obj_fcn_hess_4d,
            ineq_cnst_2d=self.ineq_cnst_2d,
            ineq_cnst_jac_3d=self.ineq_cnst_jac_3d,
            ineq_cnst_hess_4d=self.ineq_cnst_hess_4d,
            eq_cnst_2d=self.eq_cnst_2d,
            eq_cnst_jac_3d=self.eq_cnst_jac_3d,
            eq_cnst_hess_4d=self.eq_cnst_hess_4d,
        )

    def __repr__(self) -> str:
        # return json.dumps(self.to_json_data(), indent=2, default=convert_data_for_json)
        return json.dumps(self.to_json_data(), default=convert_data_for_json)
