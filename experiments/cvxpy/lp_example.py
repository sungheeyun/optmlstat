
from cvxopt import matrix, solvers
import numpy.random as nr

if __name__ == "__main__":

    A = matrix([[-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0]])
    b = matrix([1.0, -2.0, 0.0, 4.0])
    c = matrix([2.0, 1.0])

    nr.seed(701014)
    m, n = 100, 50

    A = matrix(nr.randn(m, n))
    b = matrix(nr.randn(m))
    c = matrix(nr.randn(n))

    # sol = solvers.lp(c, A, b, solver="glpk", verbose=True)
    sol = solvers.lp(c, A, b)
    # sol = solvers.lp(c, A, b, solver="mosek")

    print(sol["x"])
