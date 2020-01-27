from typing import Any

from numpy import ndarray


def convert_data_for_json(value: Any) -> Any:
    if isinstance(value, (list, dict, float, str, int)):
        return value
    elif isinstance(value, ndarray):
        return [convert_data_for_json(x_value) for x_value in value]
    else:
        return value.to_json_data()
