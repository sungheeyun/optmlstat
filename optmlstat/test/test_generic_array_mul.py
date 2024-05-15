"""

"""

import unittest

import numpy as np
from numpy.random import rand

from optmlstat.linalg.utils import generic_array_mul


class TestGenericArrayMul(unittest.TestCase):
    def test_generic_array_mul(self):
        array_1: np.ndarray = rand(12, 32)
        array_2: np.ndarray = rand(12, 32)
        array_3: np.ndarray = rand(3)
        array_4: np.ndarray = rand(4)
        array_5: np.ndarray = rand(3, 1, 4, 5)
        array_6: np.ndarray = rand(1, 3, 4, 6)
        array_7: np.ndarray = rand(3, 3, 1, 7)

        self.assertTrue(
            np.allclose(np.dot(array_1.T, array_2), generic_array_mul(array_1, array_2))
        )

        self.assertEqual(generic_array_mul(array_3, array_4).shape, (3, 4))
        self.assertTrue(
            np.allclose(
                generic_array_mul(array_3, array_4), np.dot(array_3[:, None], array_4[None, :])
            )
        )

        self.assertEqual(generic_array_mul(array_5, array_6, array_7).shape, (5, 6, 7))


if __name__ == "__main__":
    unittest.main()
