from numpy import ndarray, power


def mean_sum_squares(error_array: ndarray) -> float:
    """
    Returns the expected value of sum of squares.

    Parameters
    ----------
    error_array:
     N-by-m array where N is the number of samples and m is the number of output.

    Returns
    -------
    mean_sum_squares:
     The expected value of sum of squares.
    """
    assert error_array.ndim >= 2

    return power(error_array, 2.0).sum(axis=1).mean()
