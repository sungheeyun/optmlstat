from typing import Tuple

from numpy import ndarray, pi, sin, sqrt
from numpy.random import rand, randn

from functions.function_base import FunctionBase
from ml.stochastic_process_samplers.stochastic_process_sampler_base import (
    StochasticProcessSamplerBase,
)
from ml.predictors.simple_sinusoidal_optimal_predictor import (
    SimpleSinusoidalOptimalPredictor,
)


class SimpleSinusoidalSampler(StochasticProcessSamplerBase):
    """
    Simple sinusoidal sampler sampling the stochastic process:

      Y = sin(2*pi*X) + V

    where X is uniformly distributed in [0, 1] and V is Gaussian with zero mean.
    """

    def __init__(self, noise_variance: float) -> None:
        assert noise_variance >= 0.0, noise_variance

        self.noise_variance: float = noise_variance

    def get_number_inputs(self) -> int:
        return 1

    def get_number_outputs(self) -> int:
        return 1

    def random_sample(self, number_samples: int) -> Tuple[ndarray, ndarray]:
        x_array_2d = rand(number_samples, 1)
        y_array_2d = sin((2.0 * pi) * x_array_2d) + sqrt(
            self.noise_variance
        ) * randn(*x_array_2d.shape)

        return x_array_2d, y_array_2d

    def get_optimal_predictor(self) -> FunctionBase:
        return SimpleSinusoidalOptimalPredictor()
