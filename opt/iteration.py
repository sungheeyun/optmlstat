from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Iteration:
    outer_iter: int
    inner_iter: Optional[int] = None
