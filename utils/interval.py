from __future__ import annotations


class Interval:
    """
    One dimensional interval.

    """
    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        self.lower_bound: float = lower_bound
        self.upper_bound: float = upper_bound

    def update(self, other: Interval) -> None:
        self.lower_bound = min(self.lower_bound, other.lower_bound)
        self.upper_bound = max(self.upper_bound, other.upper_bound)
