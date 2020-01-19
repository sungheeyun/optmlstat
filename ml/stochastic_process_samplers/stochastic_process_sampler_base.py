from abc import ABC, abstractmethod
from typing import Tuple

from numpy import ndarray

from ml.predictors.predictor_base import PredictorBase


class StochasticProcessSamplerBase(ABC):
    @abstractmethod
    def get_number_inputs(self) -> int:
        pass

    @abstractmethod
    def get_number_outputs(self) -> int:
        pass

    @abstractmethod
    def random_sample(self, number_samples: int) -> Tuple[ndarray, ndarray]:
        """
        Generate random samples of (X, Y).

        Parameters
        ----------
        number_samples:
          Number of samples

        Returns
        -------
        x_train_array_2d:
          N-by-n array where N is the number of samples and n is the number of input variables.
        y_array_2d:
          N-by-m array where N is the number of samples and m is the number of output variables.
        """
        pass

    @abstractmethod
    def get_optimal_predictor(self) -> PredictorBase:
        """
        Returns an optimal predictor in least-mean-square sense.

        Returns
        -------
        predictor:
          An optimal predictor
        """
        pass
