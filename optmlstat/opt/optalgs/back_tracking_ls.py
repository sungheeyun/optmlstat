"""
back tracking line search
"""

from logging import Logger, getLogger

import numpy as np

from optmlstat.functions.function_base import FunctionBase
from optmlstat.opt.optalgs.line_search_base import LineSearchBase


logger: Logger = getLogger()


class BackTrackingLineSearch(LineSearchBase):
    def __init__(self, _alpha: float, _beta: float):
        self._alpha: float = _alpha
        self._beta: float = _beta

    def search(
        self, fcn: FunctionBase, x_array_2d: np.ndarray, search_dir_2d: np.ndarray
    ) -> np.ndarray:
        jac: np.ndarray = fcn.jacobian(x_array_2d)
        assert jac is not None
        assert jac.shape[1] == 1, jac.shape
        jac = jac.squeeze(axis=1)

        grad_search: np.ndarray = (jac * search_dir_2d).sum(axis=1, keepdims=True)

        num_pnts: int = x_array_2d.shape[0]
        t_array_1d: np.ndarray = np.zeros(num_pnts)
        active: np.ndarray = np.array([True] * num_pnts)

        y_array_2d: np.ndarray = fcn.eval(x_array_2d)

        step_len: float = 1.0
        while sum(active) > 0:
            t_array_1d[active] = step_len
            logger.debug(
                str(
                    fcn.eval(
                        x_array_2d[active] + step_len * search_dir_2d[active]
                    ).ravel()
                )
                + " > "
                + str(
                    (
                        y_array_2d[active]
                        + self._alpha * step_len * grad_search[active]
                    ).ravel()
                )
                + " ["
                + str(active)
                + "]"
            )
            active[active] = (
                fcn.eval(x_array_2d[active] + step_len * search_dir_2d[active])
                > y_array_2d[active] + self._alpha * step_len * grad_search[active]
            ).ravel()

            step_len *= self._beta

        return t_array_1d
