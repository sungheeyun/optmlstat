"""
Gaussian distribution
"""

import numpy as np

from optmlstat.stats.dists.prob_dist_base import ProbDistBase


class Gaussian(ProbDistBase):
    """
    (Multivariate) Gaussian distribution.
    This class is (supposed to be) immutable.
    """

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray | None = None,
        precision: np.ndarray | None = None,
    ) -> None:
        assert (
            covariance is None
            and precision is not None
            or covariance is not None
            and precision is None
        ), (covariance, precision)

        assert covariance is None or covariance.shape[0] == covariance.shape[1], covariance.shape
        assert covariance is None or mean.size == covariance.shape[0], (
            mean.shape,
            covariance.shape,
        )
        assert precision is None or precision.shape[0] == precision.shape[1], precision.shape
        assert precision is None or mean.size == precision.shape[0], (
            mean.shape,
            precision.shape,
        )

        self.mean: np.ndarray = mean
        self.covariance: np.ndarray | None = covariance
        self.precision: np.ndarray | None = precision

    @property
    def num_variables(self) -> int:
        return self.mean.size

    @property
    def cov_array(self) -> np.ndarray | None:
        return self.covariance

    @property
    def prc_array(self) -> np.ndarray | None:
        return self.precision
