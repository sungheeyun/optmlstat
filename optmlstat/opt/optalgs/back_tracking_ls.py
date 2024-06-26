"""
back tracking line search
"""

from logging import Logger, getLogger
from typing import Callable

import numpy as np

from optmlstat.opt.optalgs.line_search_base import LineSearchBase


logger: Logger = getLogger()


class BackTrackingLineSearch(LineSearchBase):

    def __init__(self, _alpha: float, _beta: float):
        assert 0.0 < _alpha < 0.5, _alpha
        assert 0.0 < _beta < 1.0, _beta
        self._alpha: float = _alpha
        self._beta: float = _beta

    def search(
        self,
        loss_fcn: Callable,
        x_array_2d: np.ndarray,
        search_dir_2d: np.ndarray,
        directional_deriv: np.ndarray,
    ) -> np.ndarray:

        num_members: int = x_array_2d.shape[0]
        t_array_1d: np.ndarray = np.zeros(num_members)
        active: np.ndarray = np.array([True] * num_members)

        y_array_2d: np.ndarray = loss_fcn(x_array_2d)

        step_len: float = 1.0
        while sum(active) > 0:
            t_array_1d[active] = step_len
            # logger.debug(
            #     str(loss_fcn(x_2d[active] + step_len * search_dir_2d[active]).ravel())
            #     + " > "
            #     + str((y_array_2d[active] + self._alpha * step_len * grad_search[active]).ravel())
            #     + " ["
            #     + str(active)
            #     + "]"
            # )
            active[active] = (
                loss_fcn(x_array_2d[active] + step_len * search_dir_2d[active])
                > y_array_2d[active]
                + self._alpha * step_len * directional_deriv[active][:, np.newaxis]
            ).ravel()

            step_len *= self._beta

        return t_array_1d
