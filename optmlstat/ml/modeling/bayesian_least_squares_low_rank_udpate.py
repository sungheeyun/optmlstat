from typing import Tuple

import numpy as np
import numpy.linalg as la
import scipy

from stats.dists.gaussian import Gaussian
from ml.features.feature_transformer_base import FeatureTransformerBase
from ml.features.identity_feature_transformer import IdentityFeatureTransformer
from ml.modeling.modeling_result import ModelingResult
from ml.modeling.bayesian_least_squares_base import BayesianLeastSquaresBase


class BayesianLeastSquaresLowRankUpdate(BayesianLeastSquaresBase):
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

        assert noise_precision > 0.0, noise_precision

        self.initial_prior: Gaussian = prior
        self.feature_trans: FeatureTransformerBase = feature_trans
        self.noise_precision: float = noise_precision

        self.m: np.ndarray
        self.Q: np.ndarray

        if self.initial_prior.precision is not None:
            self.m = (
                np.dot(self.initial_prior.precision, self.initial_prior.mean)
                / self.noise_precision
            )
            self.Q = self.noise_precision * la.inv(
                self.initial_prior.precision
            )
        else:
            self.m = (
                scipy.linalg.solve(
                    self.initial_prior.covariance,
                    self.initial_prior.mean,
                    assume_a="pos",
                )
                / self.noise_precision
            )
            self.Q = self.initial_prior.covariance * self.noise_precision

    def train(
        self, x_array_2d: np.ndarray, y_array_1d: np.ndarray, **kwargs
    ) -> ModelingResult:
        feature_array: np.ndarray = (
            self.feature_trans.get_transformed_features(x_array_2d)
        )  # Phi(x)

        self.m += np.dot(feature_array.T, y_array_1d)
        tmp_array_1d: np.ndarray = np.dot(self.Q, feature_array.T)
        self.Q -= np.dot(
            tmp_array_1d,
            scipy.linalg.solve(
                np.dot(feature_array, tmp_array_1d)
                + np.eye(x_array_2d.shape[0]),
                tmp_array_1d.T,
                assume_a="pos",
            ),
        )

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
        feature: np.ndarray = self.feature_trans.get_transformed_features(
            x_array_1d
        )

        tmp_array_1d: np.ndarray = np.dot(self.Q, feature)

        mean: float = np.dot(self.m, tmp_array_1d)
        variance: float = (
            np.dot(feature, tmp_array_1d) + 1.0
        ) / self.noise_precision

        return mean, variance
