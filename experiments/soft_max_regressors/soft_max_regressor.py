import abc
import typing as tp

import cvxpy as cp
from freq_used.logging_utils import set_logging_basic_config
import logging
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.linalg import lstsq
import numpy.random as nr


mpl.use("TkAgg")
logger: logging.Logger = logging.getLogger()


class Regressor(abc.ABC):
    @abc.abstractmethod
    def train(self, x_array_2d: np.ndarray, y_array_1d: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def predict(self, x_array_2d) -> np.ndarray:
        pass


class LinRegressor(Regressor):
    """
    Linear regressor
    """

    def __init__(
        self, /, *, intercept: bool = False, lambd: float = 0.0
    ) -> None:
        """
        :param intercept:
        :param lambd: the positive weight on the 2-norm regularizer
        """
        assert lambd >= 0.0, lambd
        super().__init__()

        self.intercept: bool = intercept
        self.lambd: float = lambd
        self.param_array_1d: np.ndarray = np.ndarray(0)

    def predict(self, x_array_2d: np.ndarray) -> np.ndarray:
        assert self.param_array_1d.size > 0, self.param_array_1d.shape

        x_pre_array_2d: np.ndarray = self.get_x_array_for_reg(x_array_2d)

        return np.dot(x_pre_array_2d, self.param_array_1d)

    def get_x_array_for_reg(self, x_array_2d: np.ndarray) -> np.ndarray:
        if self.intercept:
            return np.hstack((x_array_2d, np.ones((x_array_2d.shape[0], 1))))
        else:
            return x_array_2d.copy()


class RidgeRegressor(LinRegressor):
    """
    Ridge regressor
    """

    def __init__(
        self, /, *, intercept: bool = False, lambd: float = 0.0
    ) -> None:
        """
        :param lambd: positive coef for 2-norm regularizer
        """
        super().__init__(intercept=intercept, lambd=lambd)

    def train(self, x_array_2d: np.ndarray, y_array_1d: np.ndarray) -> None:
        assert x_array_2d.shape[0] == y_array_1d.size, (
            x_array_2d.shape,
            y_array_1d.shape,
        )

        x_reg_array_2d: np.ndarray = self.get_x_array_for_reg(x_array_2d)

        a_array_2d: np.ndarray = np.vstack(
            (
                x_reg_array_2d,
                np.sqrt(self.lambd) * np.eye(x_reg_array_2d.shape[1]),
            )
        )
        b_array_2d: np.ndarray = np.concatenate(
            (y_array_1d, np.zeros(x_reg_array_2d.shape[1]))
        )

        self.param_array_1d, _, _, _ = lstsq(
            a_array_2d, b_array_2d, rcond=None
        )


class PNormRegressor(LinRegressor):
    """
    Minimize p-norm of residuals.
    """

    def __init__(
        self, /, *, intercept: bool = False, lambd: float = 0.0, p_: int
    ) -> None:
        assert p_ >= 2, p_

        # super().__init__(intercept=intercept, lambd=lambd)
        super().__init__(intercept=intercept, lambd=lambd)

        self.p_: int = p_

    def train_(self, x_array_2d: np.ndarray, y_array_1d: np.ndarray) -> None:

        x_reg_array_2d: np.ndarray = self.get_x_array_for_reg(x_array_2d)

        opt_var = cp.Variable(x_reg_array_2d.shape[1])

        objective = cp.Minimize(
            cp.sum((x_reg_array_2d * opt_var - y_array_1d) ** self.p_)
        )
        problem = cp.Problem(objective)

        # problem.solve(verbose=True, solver=cp.CVXOPT)
        problem.solve(verbose=True, solver=cp.CVXOPT)

        logger.info("status: %s", problem.status)
        logger.info("optimal value: %f", problem.value)
        logger.info("optimal var: %s", opt_var.value)

        assert problem.status == cp.OPTIMAL, problem.status

        self.param_array_1d = opt_var.value

    def train__(self, x_array_2d: np.ndarray, y_array_1d: np.ndarray) -> None:

        x_reg_array_2d: np.ndarray = self.get_x_array_for_reg(x_array_2d)

        x_opt_var = cp.Variable(x_reg_array_2d.shape[1])

        objective = cp.Minimize(
            cp.sum((x_reg_array_2d * x_opt_var - y_array_1d) ** self.p_)
            + self.lambd * cp.sum_squares(x_opt_var)
        )
        problem = cp.Problem(objective)

        # problem.solve(verbose=True, solver=cp.GLPK)
        problem.solve(verbose=True)

        logger.info("status: %s", problem.status)
        logger.info("optimal value: %f", problem.value)
        logger.info("optimal var: %s", x_opt_var.value)

        assert problem.status == cp.OPTIMAL, problem.status

        self.param_array_1d = x_opt_var.value

    def train(self, x_array_2d: np.ndarray, y_array_1d: np.ndarray) -> None:

        scale_factor: float = 0.1

        x_reg_array_2d: np.ndarray = self.get_x_array_for_reg(x_array_2d)

        # x_reg_array_2d *= scale_factor
        # y_array_1d = scale_factor * y_array_1d

        x_opt_var = cp.Variable(x_reg_array_2d.shape[1])
        t_opt_var = cp.Variable(x_reg_array_2d.shape[0])

        objective = cp.Minimize(
            cp.sum((scale_factor * t_opt_var) ** self.p_)
            + scale_factor ** self.p_ * self.lambd * cp.sum_squares(x_opt_var)
        )
        cnsts = list()
        cnsts.append(x_reg_array_2d @ x_opt_var - y_array_1d <= t_opt_var)
        cnsts.append(-t_opt_var <= x_reg_array_2d @ x_opt_var - y_array_1d)

        problem = cp.Problem(objective, cnsts)

        # problem.solve(verbose=True, solver=cp.CVXOPT)
        problem.solve(
            verbose=True,
            solver=cp.ECOS,
            abstol=1e-8,
            reltol=1e-8,
            # abstol=(1e-8 * scale_factor ** self.p_),
            # reltol=(1e-8 * scale_factor ** self.p_),
        )

        logger.info("status: %s", problem.status)
        logger.info("optimal value: %f", problem.value)
        logger.info("optimal var: %s", x_opt_var.value)

        # assert problem.status == cp.OPTIMAL, problem.status

        self.param_array_1d = x_opt_var.value


class MaxRegressor(LinRegressor):
    """
    Minimize p-norm of residuals.
    """

    def train(self, x_array_2d: np.ndarray, y_array_1d: np.ndarray) -> None:

        x_reg_array_2d: np.ndarray = self.get_x_array_for_reg(x_array_2d)

        x_opt_var = cp.Variable(x_reg_array_2d.shape[1])
        t_opt_var = cp.Variable()

        objective = cp.Minimize(
            t_opt_var + self.lambd * cp.sum_squares(x_opt_var)
        )
        cnsts = list()
        cnsts.append(x_reg_array_2d @ x_opt_var - y_array_1d <= t_opt_var)
        cnsts.append(-t_opt_var <= x_reg_array_2d @ x_opt_var - y_array_1d)

        problem = cp.Problem(objective, cnsts)

        # problem.solve(verbose=True, solver=cp.CVXOPT)
        problem.solve(verbose=True)

        logger.info("status: %s", problem.status)
        logger.info("optimal value: %f", problem.value)
        logger.info("optimal var: %s", x_opt_var.value)

        assert problem.status == cp.OPTIMAL, problem.status

        self.param_array_1d = x_opt_var.value


class Error:
    P_LIST: tp.List[float] = [1.0, 2.0, 4.0, 8.0, 16.0]

    def __init__(
        self, y_array_1d: np.ndarray, y_hat_array_1d: np.ndarray
    ) -> None:
        self.y_array_1d: np.ndarray = y_array_1d
        self.y_hat_array_1d: np.ndarray = y_hat_array_1d
        self.res_array_1d: np.ndarray = self.y_array_1d - self.y_hat_array_1d
        self.abs_res_array_1d: np.ndarray = np.abs(self.res_array_1d)

        self.p_norm_errors: tp.Dict[float, float] = dict()
        self.max_error: float

        for p in self.P_LIST:
            self.p_norm_errors[p] = np.power(
                np.power(self.abs_res_array_1d, p).mean(), 1.0 / p
            )

        self.max_idx = int(self.abs_res_array_1d.argmax())
        self.max_error = self.abs_res_array_1d.max()

    def report(self) -> None:

        err1: float = self.y_array_1d.std()
        err2: float = self.y_hat_array_1d.std()

        logger.info("std of y = %f", err1)
        logger.info("std of y_hat = %f", err2)
        logger.info("std ratio = %f", (err2 / err1))
        for p in sorted(self.p_norm_errors.keys()):
            logger.info("%2d-norm error = %f", p, self.p_norm_errors[p])
        logger.info("    max error = %f", self.max_error)

    def plot_analysis(self, ax: Axes) -> None:
        ax.plot(self.y_array_1d, "bo", alpha=0.5)
        ax.plot(
            self.y_hat_array_1d, "o", mec="#FFA500", mfc="#FFA500", alpha=0.5
        )

        ax2: Axes = ax.twinx()
        ax2.plot(self.abs_res_array_1d, "x")
        ax2.plot(self.max_idx, self.abs_res_array_1d[self.max_idx], "ro")

        ax.set_title("max res = %g" % self.max_error)

        ax.plot(
            self.max_idx * np.ones(2),
            [self.y_array_1d[self.max_idx], self.y_hat_array_1d[self.max_idx]],
            "r-",
        )

        # ylim = ax.get_ylim()
        # ax.plot(self.max_idx * np.ones(2), ylim, "r-")
        # ax.set_ylim(ylim)

    def plot_res_dist(self, ax: Axes) -> None:
        ax.hist(self.res_array_1d, bins=20)


def run() -> None:
    num_data: int = 50
    num_features: int = 5
    noise_variance: float = 0.5

    nr.seed(760104)
    # nr.seed(760)

    x_array_2d: np.ndarray = nr.randn(num_data, num_features)
    w_array_1d: np.ndarray = nr.randn(num_features)
    # w2_array_1d: np.ndarray = nr.randn(num_features)

    z_array_1d: np.ndarray = np.dot(x_array_2d, w_array_1d)
    # y_array_1d += 1 * np.dot(x_array_2d ** 2, w2_array_1d)
    y_array_1d: np.ndarray = z_array_1d + np.sqrt(noise_variance) * nr.randn(
        *z_array_1d.shape
    )

    lambd: float = 0.1

    regressor_list: tp.List[Regressor] = list()
    regressor_list.append(RidgeRegressor(intercept=False, lambd=lambd))
    regressor_list.append(PNormRegressor(p_=16, lambd=lambd))
    regressor_list.append(MaxRegressor(lambd=lambd))

    for regressor in regressor_list:

        regressor.train(x_array_2d, y_array_1d)
        logger.info("opt_params: %s", regressor.param_array_1d)
        y_hat_array_1d = regressor.predict(x_array_2d)

        logger.info("%s with lambda = %g", regressor, lambd)

        err: Error = Error(y_array_1d, y_hat_array_1d)
        err.report()

        fig, (ax1, ax2) = plt.subplots(2, 1)
        err.plot_analysis(ax1)
        err.plot_res_dist(ax2)

        fig.suptitle("%s" % regressor)

        fig.show()
    plt.show()


if __name__ == "__main__":
    set_logging_basic_config(__file__)

    run()
