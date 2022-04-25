"""House price prediction.
The data is from: https://www.kaggle.com/harlfoxem/housesalesprediction
"""
import abc
import logging
from multiprocessing.pool import Pool
import sys
import typing as tp

import numpy as np
import numpy.random as nr
import networkx as nx
from networkx.classes.graph import Graph

from freq_used.logging_utils import set_logging_basic_config
import strat_models
from strat_models import Loss
from strat_models import Regularizer

# import latexify

sys.path.append("..")
logger: logging.Logger = logging.getLogger()


class Fcn(abc.ABC):
    @abc.abstractmethod
    def eval(self, x_array_1d: np.ndarray) -> float:
        pass

    @property
    @abc.abstractmethod
    def num_vars(self) -> int:
        pass

    @abc.abstractmethod
    def single_prox(
        self, t: float, nu: np.ndarray, warm_start: np.ndarray
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def sub_optimality(self, x_array_1d: np.ndarray) -> float:
        pass


class SquareSum(Fcn):
    """
    f(x) = .5 * \| x - c \|_2^2
    """

    def __init__(self, center_array_1d: np.ndarray) -> None:
        self.center: np.ndarray = center_array_1d.copy()

    def eval(self, x_array_1d: np.ndarray) -> float:
        return 0.5 * np.linalg.norm(x_array_1d - self.center) ** 2.0

    @property
    def num_vars(self) -> int:
        return self.center.size

    def single_prox(
        self, t: float, nu: np.ndarray, warm_start: np.ndarray
    ) -> np.ndarray:
        assert nu.ndim == 2, nu.shape
        assert nu.shape[1] == 1, nu.shape
        assert self.center.size == nu.shape[0], (self.center.shape, nu.shape)
        assert nu.shape == warm_start.shape, (nu.shape, warm_start.shape)

        center: np.ndarray = self.center.reshape((-1, 1))
        return (t * center + nu) / (t + 1.0)

    def sub_optimality(self, x_array_1d: np.ndarray) -> float:
        return np.linalg.norm(x_array_1d - self.center) / np.sqrt(
            self.center.size
        )


def foo(
    fcn_: Fcn, t_: float, nu_: np.ndarray, warm_start_: np.ndarray
) -> np.ndarray:
    return fcn_.single_prox(t_, nu_, warm_start_)


class ObjFcn(Loss):
    def __init__(
        self, fcn_list: tp.List[Fcn], node_idx_map: tp.Dict[tp.Any, int]
    ) -> None:
        assert fcn_list, fcn_list
        assert set(node_idx_map.values()) == set(range(len(fcn_list))), (
            sorted(node_idx_map.values()),
            len(fcn_list),
        )

        super().__init__()
        self.fcn_list: tp.List[Fcn] = fcn_list
        self.node_idx_map: tp.Dict[tp.Any, int] = node_idx_map

        self.isDistribution = False

        self.K: int = len(self.fcn_list)
        self.shape: tp.Tuple[int, int] = (self.fcn_list[0].num_vars, 1)
        self.theta_shape: tp.Tuple[int, int, int] = (self.K,) + self.shape

    def setup(self, data, G) -> tp.Dict[str, tp.Any]:
        """
        :param data:
        :param G:
        :return: cache
        """
        cache: tp.Dict[str, tp.Any] = dict(
            shape=self.shape, theta_shape=self.theta_shape
        )

        return cache

    def prox(
        self,
        t: float,
        nu: np.ndarray,
        warm_start: np.ndarray,
        pool: Pool,
        cache: tp.Dict[str, tp.Any],
    ) -> np.ndarray:
        assert nu.ndim == 3, nu.shape
        assert self.K == nu.shape[0], (self.K, nu.shape)
        assert nu.shape == warm_start.shape, (nu.shape, warm_start.shape)

        prox_list: tp.List[np.ndarray]
        num_processes: int = pool._processes
        if num_processes == 1:
            prox_list = list()
            for idx, x in enumerate(nu):
                fcn: Fcn = self.fcn_list[idx]
                prox_list.append(fcn.single_prox(t, nu[idx], warm_start[idx]))
        elif num_processes > 1:
            prox_list = pool.starmap(
                foo, zip(self.fcn_list, [t] * nu.shape[0], nu, warm_start)
            )
        else:
            assert False, num_processes

        return np.array(prox_list)

    def scores(self, data: tp.Any, graph: Graph) -> float:
        fcn_val_list: tp.List[float] = list()
        for node in graph.nodes:
            x_array_1d: np.ndarray = graph._node[node]["theta_tilde"].reshape(
                -1
            )
            fcn: Fcn = self.fcn_list[self.node_idx_map[node]]
            fcn_val_list.append(fcn.eval(x_array_1d))

        return np.sqrt(np.array(fcn_val_list, float).mean())

    def sub_optimality(self, graph: Graph) -> np.ndarray:
        sub_opt_list: tp.List[float] = list()
        for node in graph.nodes:
            x_array_1d: np.ndarray = graph._node[node]["theta_tilde"].reshape(
                -1
            )
            fcn: Fcn = self.fcn_list[self.node_idx_map[node]]
            sub_opt_list.append(fcn.sub_optimality(x_array_1d))

        return np.array(sub_opt_list, float)


def run() -> None:
    set_logging_basic_config(__file__)

    num_vars: int = 100
    num_fcns: int = 64

    obj_fcn_list: tp.List[Fcn] = list()

    nr.seed(760104)
    for _ in range(num_fcns):
        obj_fcn_list.append(SquareSum(nr.randn(num_vars)))

    graph: Graph = nx.complete_graph(len(obj_fcn_list))
    assert graph.number_of_nodes() == len(obj_fcn_list), (
        graph.number_of_nodes(),
        len(obj_fcn_list),
    )

    # assign functions to nodes
    node_idx_map: tp.Dict[tp.Any, int] = dict()
    for idx, node in enumerate(graph.nodes):
        node_idx_map[node] = idx

    strat_models.set_edge_weight(graph, 1e-6)
    # strat_models.set_edge_weight(graph, 1e-3)
    # strat_models.set_edge_weight(graph, 1e+0)
    # strat_models.set_edge_weight(graph, 1e2)

    K, n = graph.number_of_nodes(), num_vars
    logger.info(f"The stratified model will have {K * n} variables.")

    # Fit models
    kwargs = dict(
        rel_tol=1e-5,
        abs_tol=1e-5,
        maxiter=100,
        n_jobs=16,
        verbose=True,
    )

    loss: ObjFcn = ObjFcn(obj_fcn_list, node_idx_map)

    reg: Regularizer = strat_models.sum_squares_reg(lambd=0.0)

    bm = strat_models.BaseModel(loss=loss, reg=reg)

    sm_fully = strat_models.StratifiedModel(bm, graph=graph)

    info = sm_fully.fit(None, **kwargs)
    score = sm_fully.scores(None)
    logger.info("Separate model")
    logger.info(f"\tInfo = {info}")
    logger.info(f"\tLoss = {score}")

    sub_opt: np.ndarray = loss.sub_optimality(graph)
    logger.info(f"\tSub-optimality = {sub_opt.min()} - {sub_opt.max()}")


if __name__ == "__main__":
    run()
