from typing import Tuple

from numpy import ndarray, block, zeros
from numpy.linalg import solve


def strictly_convex_quadratic_with_linear_equality_constraints(
    p_array_2d: ndarray, q_array_1d: ndarray, a_array_2d: ndarray, b_array_1d: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Solve the following problem:

    minimize x^T P X + q^T x
    subject to A x = b

    where P is positive definite. This problem always has a unique solution
    IF the constraint is feasible.
    One way to solve this analytically is to use KKT condition:

    A x = b
    2 P x + q + A^T \nu = 0

    which is equivalent to

    [2P A^T] [x  ] = [-q]
    [ A 0  ] [\nu]   [ b].

    Parameters
    ----------
    p_array_2d:
     2d-array representing P
    q_array_1d:
     1d-array representing q
    a_array_2d:
     2d-array representing A
    b_array_1d:
     1d-array representing b

    Returns
    -------
    x_array_1d:
     1d-array representing optimal x
    nu_array_1d:
     1d-array representing optimal \nu
    """
    pass

    num_primal_vars: int
    num_dual_vars: int

    num_dual_vars, num_primal_vars = a_array_2d.shape

    total_num_vars: int = num_primal_vars + num_dual_vars

    kkt_a_array_2d: ndarray = block(
        [
            [p_array_2d + p_array_2d.T, a_array_2d.T],
            [a_array_2d, zeros((num_dual_vars, num_dual_vars))],
        ]
    )
    kkt_b_array_1d: ndarray = block([-q_array_1d, b_array_1d])

    assert kkt_a_array_2d.shape == (total_num_vars, total_num_vars)
    assert kkt_b_array_1d.shape == (total_num_vars,)

    solution_1d: ndarray = solve(kkt_a_array_2d, kkt_b_array_1d)

    assert solution_1d.shape == (total_num_vars,)

    return solution_1d[:num_primal_vars], solution_1d[num_primal_vars:]
