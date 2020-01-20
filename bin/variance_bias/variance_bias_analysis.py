from logging import Logger, getLogger
from typing import List, Tuple

from numpy import linspace, ndarray, array, power, newaxis, exp
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from freq_used.logging import set_logging_basic_config
from freq_used.plotting import get_figure

from functions.function_base import FunctionBase
from ml.stochastic_process_samplers.stochastic_process_sampler_base import StochasticProcessSamplerBase
from ml.stochastic_process_samplers.simple_sinusoidal_sampler import SimpleSinusoidalSampler
from ml.modeling.linear_modeler import LinearModeler
from ml.modeling.modeler_base import ModelerBase
from functions.basis_functions.gaussian_basis_function import GaussianBasisFunction
from plotting.plotting import plot_1d_data
from ml.measure import mean_sum_squares


logger: Logger = getLogger()


def variance_bias_analysis(
    stochastic_process_sampler: StochasticProcessSamplerBase,
    modeler: ModelerBase,
    x_array_2d_for_meas: ndarray,
    draw: bool = True,
) -> Tuple[float, float, float, float]:

    optimal_predictor: FunctionBase = stochastic_process_sampler.get_optimal_predictor()

    y_array_3d_prediction: ndarray = ndarray(shape=(num_samples_for_measure, 1, num_trainings))
    noise_list: List[float] = list()
    test_error_list: List[float] = list()
    for idx in range(num_trainings):
        x_train_array_2d, y_train_array_2d = stochastic_process_sampler.random_sample(training_dataset_size)
        modeler.train(x_train_array_2d, y_train_array_2d)
        predictor: FunctionBase = modeler.get_predictor()

        y_array_2d_prediction = predictor.get_y_values_2d(x_array_2d_for_meas)
        y_array_3d_prediction[:, :, idx] = y_array_2d_prediction

        noise_list.append(mean_sum_squares(optimal_predictor.get_y_values_2d(x_train_array_2d) - y_train_array_2d))

        x_test_array_2d, y_test_array_2d = stochastic_process_sampler.random_sample(test_dataset_size)
        test_error_list.append(mean_sum_squares(y_test_array_2d - predictor.get_y_values_2d(x_test_array_2d)))

    y_array_2d_sample_mean: ndarray = y_array_3d_prediction.mean(axis=2)

    y_array_2d_optimal: ndarray = optimal_predictor.get_y_values_2d(x_array_2d_for_meas)

    noise: float = array(noise_list).mean()
    bias: float = mean_sum_squares(y_array_2d_sample_mean - y_array_2d_optimal)
    variance: float = mean_sum_squares(y_array_3d_prediction - y_array_2d_sample_mean[:, newaxis])
    test_error: float = array(test_error_list).mean()

    logger.info(f"reg_coef: {reg_coef}")
    logger.info(f"noise: {noise}")
    logger.info(f"bias: {bias}")
    logger.info(f"variance: {variance}")
    logger.info(f"test_error: {test_error}")

    # plotting

    if draw:
        figure: Figure = get_figure(2, 1)

        ax1, ax2 = figure.get_axes()

        plot_1d_data(ax1, x_train_array_2d, y_train_array_2d, "o", label="random samples")
        plot_1d_data(ax1, x_array_2d_for_meas, y_array_2d_optimal, "-", label="optimal prediction")
        plot_1d_data(ax1, x_array_2d_for_meas, y_array_2d_sample_mean, "-", label="prediction mean")
        ax1.legend()

        for idx in range(min(y_array_3d_prediction.shape[2], 20)):
            plot_1d_data(ax2, x_array_2d_for_meas, y_array_3d_prediction[:, :, idx], "r-")

        for ax in figure.get_axes():
            ax.set_ylim([-1.5, 1.5])
            ax.set_xlim([0.0, 1.0])

        figure.show()

    return noise, bias, variance, test_error


if __name__ == "__main__":

    set_logging_basic_config(__file__)

    noise_variance: float = 0.01
    training_dataset_size: int = 25
    test_dataset_size: int = 25
    num_basis_functions: int = 24
    basis_function_center_array: ndarray = linspace(0, 1, num_basis_functions)
    gaussian_basis_width: float = 0.03
    num_samples_for_measure: int = 100

    num_trainings: int = 100

    # random process

    simple_sinusoidal_sampler: SimpleSinusoidalSampler = SimpleSinusoidalSampler(noise_variance)

    # measure

    x_array_2d_for_measure: ndarray = array([linspace(0.0, 1.0, num_samples_for_measure)]).T

    # basis functions

    gaussian_basis_function: GaussianBasisFunction = GaussianBasisFunction(
        [power(gaussian_basis_width, 2.0)] * basis_function_center_array.size, array([basis_function_center_array]).T
    )

    res_list: List[tuple] = list()
    log_reg_coef_array = linspace(-7.5, 2.5, 40)
    for idx, log_reg_coef in enumerate(log_reg_coef_array):
        logger.info(f"INDEX: {idx}")
        reg_coef: float = exp(log_reg_coef)
        linear_modeler: LinearModeler = LinearModeler(gaussian_basis_function, reg_coef)
        if idx % 5 == 0:
            draw = True
        else:
            draw = False
        res_list.append(variance_bias_analysis(simple_sinusoidal_sampler, linear_modeler, x_array_2d_for_measure, draw))

    data_array: ndarray = array(res_list)

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()
    ax.plot(log_reg_coef_array, data_array)
    ax.plot(log_reg_coef_array, data_array[:, :-1].sum(axis=1))
    ax.legend(["noise", "bias", "variance", "test_error", "noise + bias + variance"])
    fig.show()
