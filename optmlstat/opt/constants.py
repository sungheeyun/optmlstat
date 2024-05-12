"""
optimization constants
"""

from enum import Enum, auto


class LineSearchMethod(Enum):
    ExactLineSearch = auto()
    BackTrackingLineSearch = auto()
