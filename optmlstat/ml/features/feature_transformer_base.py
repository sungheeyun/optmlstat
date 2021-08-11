from abc import ABC, abstractmethod
import numpy as np


class FeatureTransformerBase(ABC):
    @abstractmethod
    def get_transformed_features(self, x_array_2d: np.ndarray) -> np.ndarray:
        """
        Trains the model. This can be called multiple times.

        Parameters
        ----------
        x_array_2d:
          N-by-n array for X.

        Returns
        -------
        transformed_feature:
          Transformed features
        """
