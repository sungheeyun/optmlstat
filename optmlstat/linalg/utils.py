"""
linalg utils
"""

import numpy as np
from numpy import random as nr


def get_random_pos_def_array(size: int) -> np.ndarray:
    r_array_2d: np.ndarray = nr.randn(size, size)
    return np.dot(r_array_2d, r_array_2d.T)
