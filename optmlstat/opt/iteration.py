from __future__ import annotations

from typing import Optional, List
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Iteration:
    outer_iteration: int
    inner_iteration: Optional[int] = None

    @staticmethod
    def get_outer_iteration_list(iteration_list: List[Iteration]) -> List[int]:
        return [iteration.outer_iteration for iteration in iteration_list]
