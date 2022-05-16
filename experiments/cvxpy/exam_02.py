# Solves a bounded least-squares problem.

import cvxpy as cp
import numpy as np


if __name__ == "__main__":

    # Problem data.
    m: int = 10
    n: int = 5
    np.random.seed(1)
    A: np.ndarray = np.random.randn(m, n)
    b: np.ndarray = np.random.randn(m)

    # Construct the problem.
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum((A @ x - b) ** 8) + cp.sum_squares(x))
    # objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective, constraints)

    print("Optimal value", prob.solve(verbose=True))
    print("Optimal var")
    print(x.value)  # A numpy ndarray.
