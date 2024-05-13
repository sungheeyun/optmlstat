"""
linear algebra utils
"""

import numpy as np
from numpy import random as nr
from scipy.stats import ortho_group


def get_random_pos_def_array(size_or_array_1d: int | np.ndarray) -> np.ndarray:
    res: np.ndarray
    if isinstance(size_or_array_1d, int):
        r_array_2d: np.ndarray = nr.randn(size_or_array_1d, size_or_array_1d)
        res = np.dot(r_array_2d, r_array_2d.T)
    elif isinstance(size_or_array_1d, np.ndarray):
        assert size_or_array_1d.ndim == 1, size_or_array_1d.shape
        orth_array_2d: np.ndarray = ortho_group.rvs(dim=size_or_array_1d.size)
        return np.dot(orth_array_2d, np.dot(np.diag(size_or_array_1d), orth_array_2d.T))
    else:
        assert False, size_or_array_1d.__class__

    return res
