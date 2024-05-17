"""
exceptions related to functions
"""


class UnboundedBelowException(Exception):
    pass


class UnboundedAboveException(Exception):
    pass


class InfiniteNumberOfSolutionsException(Exception):
    pass


class ValueUnknownException(Exception):
    pass
