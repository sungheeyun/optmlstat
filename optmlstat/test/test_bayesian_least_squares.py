from typing import List, Tuple
import unittest
import logging

import numpy as np
import numpy.random as nr
import scipy.stats as ss
import freq_used.logging_utils as fl

from stats.dists.gaussian import Gaussian
from ml.modeling.bayesian_least_squares_bruteforce import (
    BayesianLeastSquaresBruteforce,
)


logger: logging.Logger = logging.getLogger()


class TestBayesianLeastSquares(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fl.set_logging_basic_config(__file__)

    def test_bayesian_least_squares(self):
        input_dim: int = 100

        data_size: int = 1
        num_trainings: int = 100

        noise_precision: float = 10.0  # beta
        confidence_level: float = 0.90

        coef_1d: np.ndarray = nr.randn(input_dim)
        print(coef_1d)

        prior: Gaussian = Gaussian(
            np.zeros(input_dim), precision=np.eye(input_dim) * 4.0
        )

        bayesian_least_squares: BayesianLeastSquaresBruteforce = (
            BayesianLeastSquaresBruteforce(prior, noise_precision)
        )

        y_list: List[float] = list()
        y_hat_mean_list: List[float] = list()
        y_hat_std_list: List[float] = list()

        for idx in range(num_trainings):
            x_data_array_2d: np.ndarray = nr.randn(data_size, input_dim)

            for x_array_1d in x_data_array_2d:
                pred_dist: Tuple[
                    float, float
                ] = bayesian_least_squares.get_predictive_dist(x_array_1d)
                y_hat_mean_list.append(pred_dist[0])
                y_hat_std_list.append(np.sqrt(pred_dist[1]))

            y_data_array_1d: np.ndarray = np.dot(
                x_data_array_2d, coef_1d
            ) + nr.randn(data_size) / np.sqrt(noise_precision)
            y_list.extend(list(y_data_array_1d))

            bayesian_least_squares.train(x_data_array_2d, y_data_array_1d)

        conf_coef: float = ss.norm.ppf(0.5 * (1.0 + confidence_level))

        x_array: np.ndarray = np.arange(len(y_list))
        y_array: np.ndarray = np.array(y_list)
        y_hat_mean_array: np.ndarray = np.array(y_hat_mean_list)
        y_hat_std_array: np.ndarray = np.array(y_hat_std_list)
        lb_array: np.ndarray = y_hat_mean_array - conf_coef * y_hat_std_array
        ub_array: np.ndarray = y_hat_mean_array + conf_coef * y_hat_std_array

        within_interval_bool_array: np.ndarray = (y_array > lb_array) & (
            y_array < ub_array
        )
        print(
            within_interval_bool_array.sum() / within_interval_bool_array.size
        )

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(x_array, y_array, "o", label="observation")
        ax.plot(x_array, y_hat_mean_array, "x", label="MAP estimation")
        ax.fill_between(
            x_array,
            lb_array,
            ub_array,
            color="b",
            alpha=0.1,
            label="prediction interval",
        )
        ax.plot(
            x_array, y_hat_std_array * conf_coef, "-", label="interval size"
        )
        ax.plot(
            [x_array[0], x_array[-1]],
            conf_coef / np.sqrt(noise_precision) * np.ones(2),
            "r-",
            alpha=0.1,
            label="lower limit on prediction interval",
        )
        ax.legend()
        fig.show()

        return

        in_list: List[bool] = list()
        for idx in range(547):
            new_x_array_1d: np.ndarray = nr.rand(input_dim)
            pred_dist = bayesian_least_squares.get_predictive_dist(
                new_x_array_1d
            )

            mu, sigma = pred_dist
            std: float = np.sqrt(sigma)
            true_val: float = np.dot(
                coef_1d, new_x_array_1d
            ) + nr.randn() / np.sqrt(noise_precision)

            in_list.append(
                true_val < mu + conf_coef * std
                and true_val > mu - conf_coef * std
            )
            # print(mu - k * std , true_val, mu + k * std)

        print(sum(in_list) / len(in_list))
        print(np.array(in_list).mean())

        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
