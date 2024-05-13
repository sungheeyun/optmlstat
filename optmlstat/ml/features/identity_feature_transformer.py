import numpy as np

from optmlstat.ml.features.feature_transformer_base import FeatureTransformerBase


class IdentityFeatureTransformer(FeatureTransformerBase):
    def get_transformed_features(self, x_array_2d: np.ndarray) -> np.ndarray:
        return x_array_2d
