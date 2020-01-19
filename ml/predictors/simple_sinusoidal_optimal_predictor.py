from numpy.core._multiarray_umath import ndarray, sin, pi

from optmlstat.ml.predictors.predictor_base import PredictorBase


class SimpleSinusoidalOptimalPredictor(PredictorBase):
    """
    An optimal predictor for SimpleSinusoidalSampler in least-square-mean sense.
    """

    def get_y_values_2d(self, x_array_2d: ndarray) -> ndarray:
        y_hat_array_2d = sin((2.0 * pi) * x_array_2d)

        return y_hat_array_2d
