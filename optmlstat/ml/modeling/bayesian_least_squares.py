from typing import List, Tuple

import numpy as np
import numpy.linalg as la

from functions.function_base import FunctionBase
from stats.dists.gaussian import Gaussian
from ml.features.feature_transformer_base import FeatureTransformerBase
from ml.features.identity_feature_transformer import IdentityFeatureTransformer
from ml.modeling.modeler_base import ModelerBase
from ml.modeling.modeling_result import ModelingResult


class BayesianLeastSquares(ModelerBase):
    """
    Bayesian Least Squares
    """

    def __init__(
        self,
        prior: Gaussian,
        noise_precision: float,
        feature_trans: FeatureTransformerBase = None,
    ) -> None:
        if feature_trans is None:
            feature_trans = IdentityFeatureTransformer()

        assert prior.prc_array is not None, prior.prc_array
        assert noise_precision > 0.0, noise_precision

        self.initial_prior: Gaussian = prior
        self.feature_trans: FeatureTransformerBase = feature_trans
        self.noise_precision: float = noise_precision

        self.prior_list: List[Gaussian] = list()
        self.prior_list.append(self.initial_prior)

    def train(
        self, x_array_2d: np.ndarray, y_array_1d: np.ndarray, **kwargs
    ) -> ModelingResult:
        prior: Gaussian = self.prior_list[-1]

        feature_array: np.ndarray = (
            self.feature_trans.get_transformed_features(x_array_2d)
        )  # Phi(x)
        precision: np.ndarray = (
            prior.precision
            + self.noise_precision * np.dot(feature_array.T, feature_array)
        )  # udpate
        mean: np.ndarray = la.lstsq(
            precision,
            self.noise_precision * np.dot(y_array_1d, feature_array)
            + np.dot(prior.mean, prior.precision),
            rcond=None,
        )[0]

        posterior: Gaussian = Gaussian(mean, precision=precision)

        self.prior_list.append(posterior)

    def get_predictor(self) -> FunctionBase:
        assert False

    def get_predictive_dist(
        self, x_array_1d: np.ndarray
    ) -> Tuple[float, float]:
        """
        Returns the predictive distribution for a data point.

        Returns
        -------
        prob_dist:
          The predictive distribution.
        """
        feature: np.ndarray = x_array_1d
        posterior = self.prior_list[-1]

        mean: float = np.dot(posterior.mean, feature)
        variance: float = (
            np.dot(
                feature, la.lstsq(posterior.precision, feature, rcond=None)[0]
            )
            + 1.0 / self.noise_precision
        )

        return mean, variance
