from typing import Any

from numpy import ndarray


def ndarray_to_list(value: Any) -> Any:
    if isinstance(value, ndarray):
        return [ndarray_to_list(x_value) for x_value in value]
    else:
        return value
