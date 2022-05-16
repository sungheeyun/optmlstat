import cvxpy as cp


if __name__ == "__main__":
    # A scalar variable.
    a = cp.Variable()

    # Vector variable with shape (5,).
    x = cp.Variable(5)

    # Matrix variable with shape (5, 1).
    x = cp.Variable((5, 1))

    # Matrix variable with shape (4, 7).
    A = cp.Variable((4, 7))
