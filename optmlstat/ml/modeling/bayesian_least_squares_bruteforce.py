from typing import List, Tuple

import numpy as np
import numpy.linalg as la

from stats.dists.gaussian import Gaussian
from ml.features.feature_transformer_base import FeatureTransformerBase
from ml.features.identity_feature_transformer import IdentityFeatureTransformer
from ml.modeling.modeling_result import ModelingResult
from ml.modeling.bayesian_least_squares_base import BayesianLeastSquaresBase


class BayesianLeastSquaresBruteforce(BayesianLeastSquaresBase):
    """
    Bayesian Least Squares
    """

    def __init__(
        self,
        prior: Gaussian,
        noise_precision: float,
        feature_trans: FeatureTransformerBase = None,
        use_factorization: bool = False,
    ) -> None:
        if feature_trans is None:
            feature_trans = IdentityFeatureTransformer()

        assert prior.prc_array is not None, prior.prc_array
        assert noise_precision > 0.0, noise_precision

        self.initial_prior: Gaussian = prior
        self.feature_trans: FeatureTransformerBase = feature_trans
        self.noise_precision: float = noise_precision
        self.use_factorization: bool = use_factorization

        self.prior_list: List[Gaussian] = list()
        self.lower_tri_list: List[np.ndarray] = list()

        self.push_to_prior_list(self.initial_prior)
        if self.use_factorization:
            lower_tri: np.ndarray = la.cholesky(self.initial_prior.precision)
            self.push_to_lower_tri_list(lower_tri)

    def push_to_prior_list(self, prior: Gaussian) -> None:
        self.prior_list.append(prior)

    def push_to_lower_tri_list(self, lower_tri: np.ndarray) -> None:
        self.lower_tri_list.append(lower_tri)

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
        )  # update

        rhs: np.ndarray = self.noise_precision * np.dot(
            y_array_1d, feature_array
        ) + np.dot(prior.mean, prior.precision)

        lower_tri: np.ndarray = la.cholesky(precision)
        if self.use_factorization:
            mean: np.ndarray = (
                self.solve_linear_sys_using_lower_tri_from_chol_fac(
                    lower_tri, rhs
                )
            )
        else:
            mean: np.ndarray = la.lstsq(precision, rhs, rcond=None)[0]

        posterior: Gaussian = Gaussian(mean, precision=precision)
        self.push_to_prior_list(posterior)
        if self.use_factorization:
            self.push_to_lower_tri_list(lower_tri)

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
        posterior = self.prior_list[-1]

        mean: float = np.dot(posterior.mean, feature)
        if self.use_factorization:
            temp_array_1d: np.ndarray = (
                self.solve_linear_sys_using_lower_tri_from_chol_fac(
                    self.lower_tri_list[-1], feature
                )
            )
        else:
            temp_array_1d: np.ndarray = la.lstsq(
                posterior.precision, feature, rcond=None
            )[0]

        variance: float = (
            np.dot(feature, temp_array_1d) + 1.0 / self.noise_precision
        )

        return mean, variance
