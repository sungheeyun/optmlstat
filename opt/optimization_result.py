from typing import Optional

from numpy import ndarray

from basic_modueles.class_base import OptMLStatClassBase


class OptimizationResult(OptMLStatClassBase):
    def __init__(self) -> None:
        self.optimal_solution: Optional[ndarray] = None
