from numpy import ndarray, exp


def sigmoid(x_array: ndarray) -> ndarray:
    return 1.0 / (1.0 + exp(-x_array))
