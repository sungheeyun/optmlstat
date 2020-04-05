import unittest
from logging import Logger, getLogger

from numpy import ndarray, set_printoptions, allclose, hstack
from numpy.random import randn, seed
from numpy.linalg import eig
from freq_used.logging import set_logging_basic_config

logger: Logger = getLogger()


class TestLinAlgFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_logging_basic_config(__file__)
        set_printoptions(formatter={'float': '{: 0.3f}'.format})

    def test_upper_block_triangular(self):
        n: int = 3
        m: int = 2

        X: ndarray = randn(m + n, m + n)
        X[n:, :n] = 0.0

        logger.info(X)

        V: ndarray
        L: ndarray

        L, V = eig(X)
        logger.info(V.shape)
        logger.info(L.shape)

        logger.info(f"eigenvalues:")
        logger.info(L)
        logger.info(f"eigenvectors:")
        logger.info(V)

        l1, v1 = eig(X[:n, :n])
        l2, v2 = eig(X[n:, n:])

        logger.info(l1)
        logger.info(l2)
        logger.info(L)

        self.assertTrue(allclose(L, hstack((l1, l2))))
        logger.info(V[:n, :n])
        logger.info(v1)
        logger.info(V[0, :n] / v1[0, :] * v1)
        self.assertTrue(allclose(V[:n, :n], V[0, :n] / v1[0, :] * v1))
        logger.info(V[n:, n:])
        logger.info(v2)
        logger.info(V[n, n:])
        logger.info(v2[0, :])
        logger.info(V[n:, n:])
        logger.info(V[n, n:]/ v2[0, :] * v2)
        self.assertTrue(allclose(V[n:, n:], V[n, n:] / v2[0, :] * v2))


if __name__ == '__main__':
    unittest.main()
