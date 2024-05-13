"""
linalg utils
"""

import numpy as np
from numpy import random as nr


def get_random_pos_def_array(size_or_array_1d: int | np.ndarray) -> np.ndarray:
    res: np.ndarray
    if isinstance(size_or_array_1d, int):
        r_array_2d: np.ndarray = nr.randn(size_or_array_1d, size_or_array_1d)
        res = np.dot(r_array_2d, r_array_2d.T)
    else:
        assert False, size_or_array_1d.__class__

    return res
