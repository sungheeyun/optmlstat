import typing as tp

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import numpy.linalg as la
import numpy.random as nr

mpl.use("TkAgg")


def nn(
    x_array_1d: np.ndarray,
    w_array_2d: np.ndarray,
    c_array_1d: np.ndarray,
    w_array_1d: np.ndarray,
    c_scalar: float,
):
    # simple neural network with one hidden layer and ReLU activation function
    assert w_array_2d.shape[0] == c_array_1d.size, (
        w_array_2d.shape,
        c_array_1d.shape,
    )
    assert w_array_2d.shape[0] == w_array_1d.size, (
        w_array_2d.shape,
        w_array_1d.shape,
    )

    y_array_1d = np.dot(w_array_2d, x_array_1d) + c_array_1d
    h_array_1d = np.maximum(y_array_1d, 0)

    return np.dot(w_array_1d, h_array_1d) + c_scalar


NUM_POINTS_PER_AXIS: int = 30


def draw_surface(
    fcn: tp.Callable[[np.ndarray], float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tp.Tuple[Figure, Axes]:

    x_array: np.ndarray = np.linspace(x_min, x_max, NUM_POINTS_PER_AXIS)
    y_array: np.ndarray = np.linspace(y_min, y_max, NUM_POINTS_PER_AXIS)

    X, Y = np.meshgrid(x_array, y_array)
    Z: np.ndarray = mesh_fcn(fcn, X, Y)

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(projection="3d")

    # ax.plot_surface(X, Y, Z, alpha=0.5)
    ax.plot_wireframe(X, Y, Z, alpha=0.5)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$f(x_1, x_2)$")

    return fig, ax


def mesh_fcn(fcn: tp.Callable[[np.ndarray], float], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    assert X.ndim == 2 and Y.ndim == 2, (X.shape, Y.shape)
    assert X.shape == Y.shape, (X.shape, Y.shape)

    Z: np.ndarray = np.zeros_like(X)

    for idx_1 in range(X.shape[0]):
        for idx_2 in range(X.shape[1]):
            x_array_1d: np.ndarray = np.array([X[idx_1, idx_2], Y[idx_1, idx_2]], float)
            y_scalar: float = fcn(x_array_1d)

            Z[idx_1, idx_2] = y_scalar

    return Z


def get_dataset(num_data_points: int) -> np.ndarray:

    num_data_per_pattern: int = int(num_data_points / 4)

    xy_1: np.ndarray = np.array([[0, 0, 0.0]]).repeat(num_data_per_pattern, axis=0)
    xy_2: np.ndarray = np.array([[0, 1, 1]]).repeat(num_data_per_pattern, axis=0)
    xy_3: np.ndarray = np.array([[1, 0, 1]]).repeat(num_data_per_pattern, axis=0)
    xy_4: np.ndarray = np.array([[1, 1, 0]]).repeat(num_data_per_pattern, axis=0)

    xy_array_2d: np.ndarray = np.vstack((xy_1, xy_2, xy_3, xy_4))

    xy_array_2d += 0.1 * nr.randn(*xy_array_2d.shape)

    x_array_2d: np.ndarray = xy_array_2d[:, :2]
    y_array_1d: np.ndarray = xy_array_2d[:, 2]

    return x_array_2d, y_array_1d


def calc_se(fcn, x_array_2d: np.ndarray, y_array_1d: np.ndarray) -> float:
    """
    squared error.

    :param fcn:
    :param x_array_2d:
    :param y_array_1d:
    :return:
    """
    y_hat_array_1d: np.ndarray = np.array([fcn(x_array_1d) for x_array_1d in x_array_2d])

    res: float = np.power(y_hat_array_1d - y_array_1d, 2.0).mean()

    return res


def run():

    w_array_2d: np.ndarray = np.ones((2, 2), float)
    c_array_1d: np.ndarray = np.array([0, -1], float)
    w_array_1d: np.ndarray = np.array([1, -2], float)
    c_scalar: float = 0.0

    def foo(x_array_1d: np.ndarray) -> float:
        return nn(x_array_1d, w_array_2d, c_array_1d, w_array_1d, c_scalar)

    def quad(x_array_1d: np.ndarray) -> float:
        return la.norm(x_array_1d) ** 2.0

    # error function
    x_array_2d: np.ndarray
    y_array_1d: np.ndarray

    x_array_2d, y_array_1d = get_dataset(100)

    def nn_w(var_w_array_1d: np.ndarray) -> float:
        w_array_2d_: np.ndarray = w_array_2d.copy()
        w_array_2d_[0, :] = var_w_array_1d

        def nn_(x_array_1d: np.ndarray) -> float:
            return nn(x_array_1d, w_array_2d_, c_array_1d, w_array_1d, c_scalar)

        return calc_se(nn_, x_array_2d, y_array_1d)

    # draw XOR NN surface
    _, _ = draw_surface(quad, -3, 3, -3, 3)
    _, ax_2 = draw_surface(foo, -0.5, 1.5, -0.5, 1.5)
    _, ax_3 = draw_surface(nn_w, -1, 3, -1, 3)

    test_points: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)

    for x in test_points:
        y: float = foo(x)
        print(x, y)

        ax_2.plot([x[0]], [x[1]], [y], marker="o", color="r")

    ax_3.set_xlabel(r"$W_{1,1}$")
    ax_3.set_ylabel(r"$W_{1,2}$")
    ax_3.set_zlabel(r"$\sum_{i=1}^N (f(W,x^{(i)}) - y^{(i)})^2/N$")

    plt.show()


if __name__ == "__main__":
    run()
