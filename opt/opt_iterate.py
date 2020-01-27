from typing import Dict, Optional, Any
import json

from numpy import ndarray

from opt.opt_prob_evaluation import OptimizationProblemEvaluation
from formatting import convert_data_for_json


class OptimizationIterate:
    def __init__(
        self,
        x_array_2d: Optional[ndarray],
        primal_prob_evaluation: Optional[OptimizationProblemEvaluation] = None,
        lambda_array_2d: Optional[ndarray] = None,
        nu_array_2d: Optional[ndarray] = None,
        dual_prob_evaluation: Optional[OptimizationProblemEvaluation] = None,
    ):
        self._x_array_2d: Optional[ndarray] = x_array_2d
        self._primal_prob_evaluation: Optional[OptimizationProblemEvaluation] = primal_prob_evaluation
        self._lambda_array_2d: Optional[ndarray] = lambda_array_2d
        self._nu_array_2d: Optional[ndarray] = nu_array_2d
        self._dual_prob_evaluation: Optional[OptimizationProblemEvaluation] = dual_prob_evaluation

        if self._x_array_2d is None and self._primal_prob_evaluation is not None:
            self._x_array_2d = self._primal_prob_evaluation.x_array_2d

        if self._lambda_array_2d is None and self._dual_prob_evaluation is not None:
            self._lambda_array_2d = self._dual_prob_evaluation.x_array_2d[
                :, : self._dual_prob_evaluation.opt_prob.num_ineq_cnst
            ]

        if self._nu_array_2d is None and self._dual_prob_evaluation is not None:
            self._nu_array_2d = self._dual_prob_evaluation.x_array_2d[
                :, self._dual_prob_evaluation.opt_prob.num_ineq_cnst:
            ]

        pass

    @property
    def x_array_2d(self) -> Optional[ndarray]:
        return self._x_array_2d

    @property
    def primal_prob_evaluation(self) -> Optional[Dict[str, ndarray]]:
        return self._primal_prob_evaluation

    @property
    def lambda_array_2d(self) -> Optional[ndarray]:
        return self._lambda_array_2d

    @property
    def nu_array_2d(self) -> Optional[ndarray]:
        return self._nu_array_2d

    @property
    def dual_prob_evaluation(self) -> Optional[Dict[str, Any]]:
        return self._dual_prob_evaluation

    def to_json_data(self) -> Dict[str, Any]:
        return dict(
            x_array_2d=self.x_array_2d,
            primal_prob_evaluation=self.primal_prob_evaluation,
            lambda_array_2d=self.lambda_array_2d,
            nu_array_2d=self.nu_array_2d,
            dual_prob_evaluation=self.dual_prob_evaluation,
        )

    def __repr__(self) -> str:
        return json.dumps(self.to_json_data(), indent=2, default=convert_data_for_json)
