"""
test block array function
"""

import unittest

import numpy as np
from numpy.random import randn

from optmlstat.linalg.utils import block_array


class TestBlockArray(unittest.TestCase):
    def test_block_array_1d_and_2d(self):
        a: np.ndarray = randn(2, 2)
        b: np.ndarray = randn(3)
        c: float = 2.0
        d: np.ndarray = randn(5, 3)

        self.assertTrue(np.allclose(block_array([c, b]), np.hstack((c, b))))
        self.assertTrue(np.allclose(block_array([b, c]), np.array([b[0], b[1], b[2], c])))

        self.assertTrue(
            np.allclose(
                block_array([[a, b], [c, d]]),
                np.concatenate(
                    (
                        np.concatenate((a, b + np.zeros((2, 3))), axis=1),
                        np.concatenate((c * np.ones((5, 2)), d), axis=1),
                    ),
                    axis=0,
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                block_array([[a, b], [c, d]]),
                np.vstack(
                    (np.hstack((a, b + np.zeros((2, 3)))), np.hstack((c * np.ones((5, 2)), d)))
                ),
            )
        )

    def test_block_array_3d(self):
        """
        test - dims - [[2, 3], [4, 5, 6], [7, 8]]
        """

        a: np.ndarray = randn(2, 4, 7)
        b: np.ndarray = randn(2, 4, 8)
        c: np.ndarray = randn(2, 5, 7)
        d: np.ndarray = randn(2, 5, 8)
        e: np.ndarray = randn(2, 6, 7)
        f: np.ndarray = randn(2, 6, 8)

        g: np.ndarray = randn(3, 4, 7)
        h: np.ndarray = randn(3, 4, 8)
        i: np.ndarray = randn(3, 5, 7)
        j: np.ndarray = randn(3, 5, 8)
        k: np.ndarray = randn(3, 6, 7)
        l: np.ndarray = randn(3, 6, 8)

        self.assertEqual(
            block_array([[[a, b], [c, d], [e, f]], [[g, h], [i, j], [k, l]]]).shape, (5, 15, 15)
        )

        def test_block_array_3d_partial(self):
            """
            test - dims - [[2, 3], [4, 5, 6], [7, 8]]
            """

            a: np.ndarray = randn(2, 4, 7)
            b: np.ndarray = randn(8)
            c: np.ndarray = randn(2, 5, 7)
            d: np.ndarray = randn(8)
            e: np.ndarray = randn(2, 6, 7)
            f: int = -3

            g: np.ndarray = randn(3, 4, 7)
            h: np.ndarray = randn(3, 4, 8)
            i: np.ndarray = randn(7)
            j: np.ndarray = randn(3, 5, 8)
            k: float = 8.0
            l: np.ndarray = randn(3, 6, 8)

            self.assertEqual(
                block_array([[[a, b], [c, d], [e, f]], [[g, h], [i, j], [k, l]]]).shape, (5, 15, 15)
            )

    def test_block_array_3d_error(self):
        """
        test - dims - [[2, 3], [4, 5, 6], [7, 8]]
        """

        try:
            a: np.ndarray = randn(2, 4, 7)
            b: np.ndarray = randn(8)
            c: np.ndarray = randn(2, 5, 7)
            d: np.ndarray = randn(8)
            e: np.ndarray = randn(2, 6, 7)
            f: int = -3

            g: np.ndarray = randn(3, 4, 7)
            h: np.ndarray = randn(4, 8)
            i: np.ndarray = randn(6)
            j: np.ndarray = randn(3, 5, 8)
            k: float = 8.0
            l: np.ndarray = randn(3, 6, 8)

            self.assertEqual(
                block_array([[[a, b], [c, d], [e, f]], [[g, h], [i, j], [k, l]]]).shape, (5, 15, 15)
            )
        except AssertionError:
            pass
        else:
            self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
