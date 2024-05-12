"""
optimization iterator
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Iteration:
    outer_iteration: int
    inner_iteration: int | None = None

    @staticmethod
    def get_outer_iteration_list(iteration_list: list[Iteration]) -> list[int]:
        return [iteration.outer_iteration for iteration in iteration_list]
