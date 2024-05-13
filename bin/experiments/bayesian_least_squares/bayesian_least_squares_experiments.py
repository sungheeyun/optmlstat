"""
experiments for Bayesian LS
"""

from typing import List, Tuple, TypeVar, Type
import logging
import time

import matplotlib.pyplot as plt
import tqdm
import freq_used.plotting as fp
import numpy as np
import numpy.random as nr
import numpy.linalg as la
import scipy.stats as ss
import freq_used.logging_utils as fl
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from optmlstat.stats.dists.gaussian import Gaussian
from optmlstat.ml.modeling.bayesian_least_squares_base import BayesianLeastSquaresBase
from optmlstat.ml.modeling.bayesian_least_squares_bruteforce import BayesianLeastSquaresBruteforce
from optmlstat.ml.modeling.bayesian_least_squares_standard import BayesianLeastSquaresStandard
from optmlstat.ml.modeling.bayesian_least_squares_low_rank_udpate import (
    BayesianLeastSquaresLowRankUpdate,
)

logger: logging.Logger = logging.getLogger()

U = TypeVar("U", bound=BayesianLeastSquaresBase)


def test_bayesian_least_squares(bayesian_ls_cls: Type[U]) -> None:
    input_dim: int = 500

    batch_size: int = 100
    num_trainings: int = 5

    logger.info(f"input dimension: {input_dim}")
    logger.info(f"batch size: {batch_size}")
    logger.info(f"# trainings: {num_trainings}")

    noise_precision: float = 10.0  # beta
    confidence_level: float = 0.9

    logger.info(f"noise_precision or beta: {noise_precision}")
    logger.info(f"confidence leveL: {confidence_level}")

    coef_1d: np.ndarray = nr.randn(input_dim)
    logger.debug(f"coefficient for the linear model: {coef_1d}")

    logger.info("Assign the prior Gaussian.")
    prior: Gaussian = Gaussian(np.zeros(input_dim), precision=np.eye(input_dim))

    logger.info("Instantiate the modeling class.")
    bayesian_least_squares: BayesianLeastSquaresBase = bayesian_ls_cls(
        prior, noise_precision
    )  # type:ignore

    y_list: List[float] = list()
    y_hat_mean_list: List[float] = list()
    y_hat_std_list: List[float] = list()

    training_time_list: List[float] = list()
    inf_time_list: List[float] = list()

    T0: float = time.time()
    prior_list: List[Gaussian] = list()
    prior_list.append(bayesian_least_squares.get_prior())
    for idx in tqdm.tqdm(range(num_trainings)):
        logger.debug(f"randomly generate {batch_size} data points")
        x_data_array_2d: np.ndarray = nr.randn(batch_size, input_dim)

        logger.debug(f"calculate predictive distribution for the {batch_size}" " data points")
        inf_time: float = 0.0
        for x_array_1d in x_data_array_2d:
            t0 = time.time()
            pred_dist: Tuple[float, float] = bayesian_least_squares.get_predictive_dist(x_array_1d)
            inf_time += time.time() - t0
            y_hat_mean_list.append(pred_dist[0])
            y_hat_std_list.append(np.sqrt(pred_dist[1]))
        inf_time_list.append(inf_time)

        y_data_array_1d: np.ndarray = np.dot(x_data_array_2d, coef_1d) + nr.randn(
            batch_size
        ) / np.sqrt(noise_precision)
        y_list.extend(list(y_data_array_1d))

        logger.debug(
            f"{idx + 1}th training with {batch_size} data points, i.e., " "updating the prior"
        )
        t0 = time.time()
        bayesian_least_squares.train(x_data_array_2d, y_data_array_1d)
        training_time_list.append(time.time() - t0)

        prior_list.append(bayesian_least_squares.get_prior())

    logger.info(f"total process took {time.time() - T0:g} sec.")

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

    # predictive distribution
    y_hat_mean_array: np.ndarray = np.array(y_hat_mean_list)
    y_hat_std_array: np.ndarray = np.array(y_hat_std_list)

    # residuals
    res_array: np.ndarray = y_array - y_hat_mean_array

    # confidence_level prediction interval
    conf_half_int: np.ndarray = conf_coef * y_hat_std_array
    ub_array: np.ndarray = y_hat_mean_array + conf_half_int
    lb_array: np.ndarray = y_hat_mean_array - conf_half_int

    within_interval_bool_array: np.ndarray = (y_array > lb_array) & (y_array < ub_array)
    logger.info(within_interval_bool_array.sum() / within_interval_bool_array.size)

    moving_average_window_size: int = min(100, int(within_interval_bool_array.size / 2))
    logger.info(f"moving_average_window_size: {moving_average_window_size}")
    moving_average_filter: np.ndarray = (
        np.ones(moving_average_window_size) / moving_average_window_size
    )
    moving_average_array: np.ndarray = np.convolve(
        within_interval_bool_array, moving_average_filter, mode="valid"
    )

    # display results
    alpha: float = 0.5
    fig_list: List[Figure] = list()
    ax_1: Axes
    ax_2: Axes
    ax_3: Axes

    fig_1, ax_1 = plt.subplots()
    fig_2, ax_2 = plt.subplots()
    fig_3, ax_3 = plt.subplots()

    fig_list.append(fig_1)
    fig_list.append(fig_2)
    fig_list.append(fig_3)

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
    ax_1.legend()

    x_lim = ax_1.get_xlim()

    ax_2.plot(x_array, res_array, label="residuals")
    ax_2.plot(x_array, conf_half_int, "r-", alpha=alpha, label="error ub")
    ax_2.plot(x_array, -conf_half_int, "r-", alpha=alpha, label="error lb")
    ax_2.plot(
        [x_array[0], x_array[-1]],
        conf_coef / np.sqrt(noise_precision) * np.ones(2),
        "k-",
        alpha=0.9,
        label="limit on prediction interval",
    )
    ax_2.plot(
        [x_array[0], x_array[-1]],
        -conf_coef / np.sqrt(noise_precision) * np.ones(2),
        "k-",
        alpha=0.9,
        label="limit on prediction interval",
    )
    ax_2.set_xlim(x_lim)
    ax_2.legend()

    ax_3.plot(
        x_array[x_array.size - moving_average_array.size :],  # noqa: #203
        moving_average_array,
        label=f"prob estimation using past {moving_average_window_size}" " points",
    )

    ax_3.plot(
        [-x_array.size, x_array.size * 2],
        np.ones(2) * confidence_level,
        "r-",
        alpha=0.5,
    )
    ax_3.set_xlim(x_lim)
    ax_3.legend()

    # inspect the changes of prediction intervals along with a line
    line_a: np.ndarray = nr.randn(input_dim)
    line_a = line_a / la.norm(line_a)
    # line_b: np.ndarray = np.zeros_like(line_a)
    line_b: np.ndarray = nr.randn(input_dim)
    line_b = line_b / la.norm(line_b)

    num_points: int = 100

    fig: Figure = fp.get_figure(
        len(prior_list),
        1,
        axis_width=5,
        axis_height=0.8,
        vertical_padding=0.1,
        top_margin=0.5,
        bottom_margin=0.5,
        left_margin=0.5,
        right_margin=0.5,
    )
    axes_list: List[Axes] = fig.get_axes()
    fig_list.append(fig)

    t_array: np.ndarray = np.linspace(-2.0, 2.0, num_points)

    for idx, prior in enumerate(prior_list):
        axis = axes_list[idx]
        coef_a: float = np.dot(prior.mean, line_a)
        coef_b: float = np.dot(prior.mean, line_b)

        assert prior.covariance is not None
        coef_c2: float = np.dot(np.dot(prior.covariance, line_a), line_a)
        coef_c1: float = 2.0 * np.dot(np.dot(prior.covariance, line_a), line_b)
        coef_c0: float = (
            np.dot(np.dot(prior.covariance, line_b), line_b)
            + 1.0 / bayesian_least_squares.noise_precision  # type:ignore
        )

        post_mean: np.ndarray = coef_a * t_array + coef_b
        post_std: np.ndarray = coef_c2 * np.power(t_array, 2.0) + coef_c1 * t_array + coef_c0
        post_lb: np.ndarray = post_mean - conf_coef * post_std
        post_ub: np.ndarray = post_mean + conf_coef * post_std

        axis.plot(t_array, post_mean, label="posterior mean")
        axis.fill_between(
            t_array,
            post_lb,
            post_ub,
            color="r",
            label="prediction interval",
            alpha=0.5,
        )
        axis.set_xlim((-1, 1))
        axis.set_ylim((-3.5, 3.5))
        axis.legend()

    for fig in fig_list:
        fig.show()
    # fig_list[-1].show()


if __name__ == "__main__":
    if "__file__" in dir():
        fl.set_logging_basic_config(__file__)

    nr.seed(760104)

    try:
        test_bayesian_least_squares(BayesianLeastSquaresBruteforce)
    except AttributeError:
        pass

    try:
        test_bayesian_least_squares(BayesianLeastSquaresStandard)
    except AttributeError:
        pass

    test_bayesian_least_squares(BayesianLeastSquaresLowRankUpdate)

    if "__file__" in dir():
        pass
        # plt.show()
