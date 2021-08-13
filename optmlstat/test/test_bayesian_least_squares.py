from typing import List, Tuple
import unittest
import logging
import time

import freq_used.plotting as fp
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
        # nr.seed(760104)

    def test_bayesian_least_squares(self):
        input_dim: int = 300

        batch_size: int = 10
        num_trainings: int = 73

        logger.info(f"input dimension: {input_dim}")
        logger.info(f"batch size: {batch_size}")
        logger.info(f"# trainings: {num_trainings}")

        noise_precision: float = 10.0  # beta
        confidence_level: float = 0.9

        logger.info(f"noise_precision or beta: {noise_precision}")
        logger.info(f"confidence leveL: {confidence_level}")

        coef_1d: np.ndarray = nr.randn(input_dim)
        # print(coef_1d)

        logger.info("Assign the prior Gaussian.")
        prior: Gaussian = Gaussian(
            np.zeros(input_dim), precision=np.eye(input_dim)
        )

        logger.info("Instantiate the modeling class.")
        bayesian_least_squares: BayesianLeastSquaresBruteforce = (
            BayesianLeastSquaresBruteforce(
                prior, noise_precision, use_factorization=False
            )
        )

        y_list: List[float] = list()
        y_hat_mean_list: List[float] = list()
        y_hat_std_list: List[float] = list()

        training_time_list: List[float] = list()
        inf_time_list: List[float] = list()

        T0: float = time.time()
        for idx in range(num_trainings):
            # logger.info(f"randomly generate {batch_size} data points")
            x_data_array_2d: np.ndarray = nr.randn(batch_size, input_dim)

            # logger.info(
            #    f"calculate predictive distribution for the {batch_size}"
            #    " data points"
            # )
            inf_time: float = 0.0
            for x_array_1d in x_data_array_2d:
                t0 = time.time()
                pred_dist: Tuple[
                    float, float
                ] = bayesian_least_squares.get_predictive_dist(x_array_1d)
                inf_time += time.time() - t0
                y_hat_mean_list.append(pred_dist[0])
                y_hat_std_list.append(np.sqrt(pred_dist[1]))
            inf_time_list.append(inf_time)

            y_data_array_1d: np.ndarray = np.dot(
                x_data_array_2d, coef_1d
            ) + nr.randn(batch_size) / np.sqrt(noise_precision)
            y_list.extend(list(y_data_array_1d))

            logger.info(
                f"{idx + 1}th training with {batch_size} data points, i.e., "
                "updating the prior"
            )
            t0: float = time.time()
            bayesian_least_squares.train(x_data_array_2d, y_data_array_1d)
            training_time_list.append(time.time() - t0)

        logger.info(f"total process took {time.time()-T0:g} sec.")

        logger.info(
            f"total training ({num_trainings} trainings with {batch_size}"
            f" as batch size) took {sum(training_time_list):g} sec."
        )
        logger.info(
            f"total inferencing ({num_trainings} trainings with {batch_size}"
            f" as batch size) took {sum(inf_time_list):g} sec."
        )

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

        moving_average_window_size: int = min(
            100, within_interval_bool_array.size / 2
        )
        moving_average_filter: np.ndarray = (
            np.ones(moving_average_window_size) / moving_average_window_size
        )
        moving_average_array: np.ndarray = np.convolve(
            within_interval_bool_array, moving_average_filter, mode="valid"
        )

        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        fig: Figure = fp.get_figure(2, 1, axis_width=6, axis_height=3)
        ax_1: Axes
        ax_2: Axes
        ax_1, ax_2 = fig.get_axes()

        ax_1.plot(x_array, y_array, "o", label="observation")
        ax_1.plot(x_array, y_hat_mean_array, "x", label="MAP estimation")
        ax_1.fill_between(
            x_array,
            lb_array,
            ub_array,
            color="b",
            alpha=0.1,
            label="prediction interval",
        )
        ax_1.plot(
            x_array, y_hat_std_array * conf_coef, "-", label="interval size"
        )
        ax_1.plot(
            [x_array[0], x_array[-1]],
            conf_coef / np.sqrt(noise_precision) * np.ones(2),
            "r-",
            alpha=0.1,
            label="lower limit on prediction interval",
        )
        ax_1.legend()

        ax_2.plot(
            x_array[x_array.size - moving_average_array.size:],
            moving_average_array,
        )
        ax_2.plot(
            [-x_array.size, x_array.size * 2],
            np.ones(2) * confidence_level,
            "r-",
            alpha=0.5,
        )
        ax_2.set_xlim(ax_1.get_xlim())
        ax_2.set_ylim((0, 1))
        fig.show()

        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
