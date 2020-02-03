from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Iteration:
    outer_iteration: int
    inner_iteration: Optional[int] = None
