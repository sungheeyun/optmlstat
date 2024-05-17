"""
optimization iterate
"""

from dataclasses import dataclass
from typing import Any
import json

import numpy as np
from numpy import ndarray

from optmlstat.opt.opt_prob import OptProb
from optmlstat.opt.opt_prob_eval import OptProbEval
from optmlstat.formatting import convert_data_for_json


@dataclass(frozen=True)
class OptimizationIterate:
    primal_prob_evaluation: OptProbEval
    dual_prob_evaluation: OptProbEval
    terminated: np.ndarray

    @property
    def x_array_2d(self) -> ndarray:
        return self.primal_prob_evaluation.x_array_2d

    @property
    def lambda_array_2d(self) -> ndarray:
        opt_prob: OptProb = self.dual_prob_evaluation.opt_prob
        return self.dual_prob_evaluation.x_array_2d[:, : opt_prob.num_ineq_cnst]

    @property
    def nu_array_2d(self) -> ndarray:
        return self.dual_prob_evaluation.x_array_2d[
            :, self.primal_prob_evaluation.opt_prob.num_ineq_cnst :  # noqa:E203
        ]

    def to_json_data(self) -> dict[str, Any]:
        return dict(
            primal_prob_evaluation=self.primal_prob_evaluation.to_json_data(),
            dual_prob_evaluation=self.dual_prob_evaluation.to_json_data(),
        )

    def __repr__(self) -> str:
        return json.dumps(self.to_json_data(), indent=2, default=convert_data_for_json)
