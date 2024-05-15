"""
test idx_iterator
"""

import unittest

from optmlstat.linalg.utils import idx_iterator


class TestIdxIterator(unittest.TestCase):
    def test_idx_iterator(self):
        ans: list[tuple[int, int, int]] = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3),
            (0, 1, 0),
            (0, 1, 1),
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 0),
            (0, 2, 1),
            (0, 2, 2),
            (0, 2, 3),
            (1, 0, 0),
            (1, 0, 1),
            (1, 0, 2),
            (1, 0, 3),
            (1, 1, 0),
            (1, 1, 1),
            (1, 1, 2),
            (1, 1, 3),
            (1, 2, 0),
            (1, 2, 1),
            (1, 2, 2),
            (1, 2, 3),
        ]

        self.assertEqual(idx_iterator((2, 3, 4)), ans)
        # print(idx_iterator((2, 3, 4)))
        self.assertEqual(idx_iterator((3, 2)), [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
        # print(idx_iterator((3, 2)))
        self.assertEqual(idx_iterator((3,)), [(0,), (1,), (2,)])
        # print(idx_iterator((3,)))


if __name__ == "__main__":
    unittest.main()
