import cvxpy as cp

if __name__ == "__main__":

    # Create two scalar optimization variables.
    x = cp.Variable()
    y = cp.Variable()

    # Create two constraints.
    constraints = [x + y == 1, x - y >= 1]

    # Form objective.
    obj = cp.Minimize((x - y) ** 2)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True)  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", x.value, y.value)
