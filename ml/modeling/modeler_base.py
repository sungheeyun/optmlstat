from abc import ABC, abstractmethod

from numpy import ndarray

from ml.modeling.modeling_result import ModelingResult
from ml.predictors.predictor_base import PredictorBase


class ModelerBase(ABC):
    """
    Base class for modelers

    """

    @abstractmethod
    def train(self, x_array_2d: ndarray, y_array_2d, **kwargs) -> ModelingResult:
        """
        Trains the model. This can be called multiple times.

        Parameters
        ----------
        x_array_2d:
          N-by-n array for X.
        y_array_2d:
          N-by-m array for Y.

        Returns
        -------
        modeling_result:
          Result of modeling. For example, number of iterations used, training error, exit status, etc.
        """
        pass

    @abstractmethod
    def get_predictor(self) -> PredictorBase:
        """
        Returns the resulting predictor obtained from the tranings so far.

        Returns
        -------
        predictor:
          The best predictor found so far.
        """
        pass
