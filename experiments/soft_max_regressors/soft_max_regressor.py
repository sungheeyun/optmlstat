"""

"""

import abc
import logging
import typing as tp

import cvxpy as cp
import matplotlib as mpl
import numpy as np
import numpy.random as nr
from freq_used.logging_utils import set_logging_basic_config
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axes._base import _AxesBase
from numpy.linalg import lstsq

mpl.use("TkAgg")
logger: logging.Logger = logging.getLogger()


class FeatureTransformer(abc.ABC):
    @abc.abstractmethod
    def transform(self, x_array_2d: np.ndarray) -> np.ndarray:
        pass


class IdentityFeatureTransformer(FeatureTransformer):
    def transform(self, x_array_2d: np.ndarray) -> np.ndarray:
        return x_array_2d.copy()


class QuadFeatureTransformer(FeatureTransformer):
    def __init__(self, /, *, square: bool = True, cross_product: bool = True) -> None:
        super().__init__()
        self.square: bool = square
        self.cross_product: bool = cross_product

    def transform(self, x_array_2d: np.ndarray) -> np.ndarray:
        array_list: tp.List[np.ndarray] = [x_array_2d]

        if self.square:
            array_list.append(x_array_2d**2)

        if self.cross_product:
            array_list.append(self.get_cross_product_array(x_array_2d))

        return np.hstack(array_list)

    @staticmethod
    def get_cross_product_array(x_array_2d: np.ndarray) -> np.ndarray:
        num_features: int = x_array_2d.shape[1]
        array_list: tp.List = list()

        for i1 in range(num_features - 1):
            array_list.append(x_array_2d[:, i1].reshape((-1, 1)) * x_array_2d[:, i1 + 1 :])

        return np.hstack(array_list)


class Regressor(abc.ABC):
    def __init__(self, feature_transformer: FeatureTransformer | None = None) -> None:
        if feature_transformer is None:
            feature_transformer = IdentityFeatureTransformer()
        self.feature_transformer: FeatureTransformer = feature_transformer

        self.param_array_1d: np.ndarray = np.ndarray(0)

    @abc.abstractmethod
    def train(self, x_array_2d: np.ndarray, y_array_1d: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def predict(self, x_array_2d) -> np.ndarray:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass


class LinRegressor(Regressor):
    """
    Linear regressor
    """

    def __init__(self, /, *, intercept: bool = False, lambd: float = 0.0, **kwargs) -> None:
        """
        :param intercept:
        :param lambd: the positive weight on the 2-norm regularizer
        """
        assert lambd >= 0.0, lambd
        super().__init__(**kwargs)

        self.intercept: bool = intercept
        self.lambd: float = lambd

    def predict(self, x_array_2d: np.ndarray) -> np.ndarray:
        assert self.param_array_1d.size > 0, self.param_array_1d.shape

        x_pre_array_2d: np.ndarray = self.get_x_array_for_reg(x_array_2d)

        return np.dot(x_pre_array_2d, self.param_array_1d)

    def get_x_array_for_reg(self, x_trans_array_2d: np.ndarray) -> np.ndarray:
        x_trans_array_2d = self.feature_transformer.transform(x_trans_array_2d)
        if self.intercept:
            return np.hstack((x_trans_array_2d, np.ones((x_trans_array_2d.shape[0], 1))))
        else:
            return x_trans_array_2d


class RidgeRegressor(LinRegressor):
    """
    Ridge regressor
    """

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
        b_array_2d: np.ndarray = np.concatenate((y_array_1d, np.zeros(x_reg_array_2d.shape[1])))

        self.param_array_1d, _, _, _ = lstsq(a_array_2d, b_array_2d, rcond=None)

    def __repr__(self) -> str:
        # return f"ridge regressor with lambda = {self.lambd:g}"
        return "ridge regressor"


class PNormRegressor(LinRegressor):
    """
    Minimize p-norm of residuals.
    """

    def __init__(
        self,
        /,
        *,
        intercept: bool = False,
        lambd: float = 0.0,
        p_: int,
        **kwargs,
    ) -> None:
        assert p_ >= 2, p_

        # super().__init__(intercept=intercept, lambd=lambd)
        super().__init__(intercept=intercept, lambd=lambd, **kwargs)

        self.p_: int = p_

    def train_(self, x_array_2d: np.ndarray, y_array_1d: np.ndarray) -> None:
        x_reg_array_2d: np.ndarray = self.get_x_array_for_reg(x_array_2d)

        opt_var = cp.Variable(x_reg_array_2d.shape[1])

        objective = cp.Minimize(cp.sum((x_reg_array_2d * opt_var - y_array_1d) ** self.p_))
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
            + scale_factor**self.p_ * self.lambd * cp.sum_squares(x_opt_var)
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

    def __repr__(self) -> str:
        return f"p-norm regressor with p = {self.p_}"


class MaxRegressor(LinRegressor):
    """
    Minimize p-norm of residuals.
    """

    def train(self, x_array_2d: np.ndarray, y_array_1d: np.ndarray) -> None:
        x_reg_array_2d: np.ndarray = self.get_x_array_for_reg(x_array_2d)

        x_opt_var = cp.Variable(x_reg_array_2d.shape[1])
        t_opt_var = cp.Variable()

        objective = cp.Minimize(
            t_opt_var
            + self.lambd
            # t_opt_var + self.lambd * cp.sum_squares(x_opt_var)
        )
        cnsts = list()
        cnsts.append(x_reg_array_2d @ x_opt_var - y_array_1d <= t_opt_var)
        cnsts.append(-t_opt_var <= x_reg_array_2d @ x_opt_var - y_array_1d)

        problem = cp.Problem(objective, cnsts)

        problem.solve(verbose=True, solver=cp.CVXOPT)
        # problem.solve(verbose=True)

        logger.info("status: %s", problem.status)
        logger.info("optimal value: %f", problem.value)
        logger.info("optimal var: %s", x_opt_var.value)

        assert problem.status == cp.OPTIMAL, problem.status

        self.param_array_1d = x_opt_var.value

    def __repr__(self) -> str:
        return "max regressor"


class ModelError:
    P_LIST: tp.List[float] = [1.0, 2.0, 4.0, 8.0, 16.0]

    def __init__(self, y_array_1d: np.ndarray, y_hat_array_1d: np.ndarray) -> None:
        self.y_array_1d: np.ndarray = y_array_1d
        self.y_hat_array_1d: np.ndarray = y_hat_array_1d
        self.res_array_1d: np.ndarray = self.y_array_1d - self.y_hat_array_1d
        self.abs_res_array_1d: np.ndarray = np.abs(self.res_array_1d)

        self.p_norm_errors: tp.Dict[float, float] = dict()
        self.max_error: float

        for p in self.P_LIST:
            self.p_norm_errors[p] = np.power(np.power(self.abs_res_array_1d, p).mean(), 1.0 / p)

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
        ax.plot(self.y_hat_array_1d, "o", mec="#FFA500", mfc="#FFA500", alpha=0.5)

        ax2: _AxesBase = ax.twinx()
        assert isinstance(ax2, Axes), ax2.__class__
        ax2.plot(self.abs_res_array_1d, "x")
        ax2.plot(self.max_idx, self.abs_res_array_1d[self.max_idx], "ro")

        ax2.set_ylim((0.0, 2.0))

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


class ModelErrorContainer:
    def __init__(self) -> None:
        self.model_name_list: tp.List[str] = list()
        self.model_error_list: tp.List[ModelError] = list()

    def put_model_error(self, model_name: str, model_error: ModelError) -> None:
        self.model_name_list.append(model_name)
        self.model_error_list.append(model_error)

    def report(self, model_name_list: tp.Optional[tp.List[str]] = None) -> None:
        if (self.model_name_list) == 0:
            return

        if model_name_list is None:
            model_name_list = self.model_name_list

        p_set: tp.Set[float] = set(self.model_error_list[0].p_norm_errors)
        for model_error in self.model_error_list:
            assert set(model_error.p_norm_errors) == p_set, (
                set(model_error.p_norm_errors),
                p_set,
            )

        logger.info(
            "                 %s",
            "".join([f"{model_name:>10}" for model_name in model_name_list]),
        )
        for p_ in sorted(p_set):
            logger.info(
                "%2d-norm error = %s",
                p_,
                "".join(
                    [
                        f"{model_error.p_norm_errors[p_]:10.3g}"
                        for model_error in self.model_error_list
                    ]
                ),
            )
        logger.info(
            "    max error = %s",
            "".join([f"{model_error.max_error:10.3g}" for model_error in self.model_error_list]),
        )


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
    y_array_1d: np.ndarray = z_array_1d + np.sqrt(noise_variance) * nr.randn(*z_array_1d.shape)

    lambd: float = 0.1

    kwargs = dict(lambd=lambd)
    kwargs.update(feature_transformer=QuadFeatureTransformer())  # type:ignore

    regressor_list: tp.List[Regressor] = list()
    regressor_list.append(RidgeRegressor(intercept=False, **kwargs))
    regressor_list.append(PNormRegressor(p_=8, **kwargs))  # type:ignore
    regressor_list.append(MaxRegressor(**kwargs))  # type:ignore

    model_err_container: ModelErrorContainer = ModelErrorContainer()

    for regressor in regressor_list:
        regressor.train(x_array_2d, y_array_1d)
        logger.info("opt_params: %s", regressor.param_array_1d)
        y_hat_array_1d = regressor.predict(x_array_2d)

        logger.info("%s with lambda = %g", regressor, lambd)

        model_err: ModelError = ModelError(y_array_1d, y_hat_array_1d)
        model_err.report()
        model_err_container.put_model_error("", model_err)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 10.0))
        model_err.plot_analysis(ax1)
        model_err.plot_res_dist(ax2)

        fig.suptitle("%s" % regressor)

        fig.show()

    model_err_container.report(["ridge", "p-norm", "max-norm"])

    plt.show()


if __name__ == "__main__":
    set_logging_basic_config(__file__)

    run()
