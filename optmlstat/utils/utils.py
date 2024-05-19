"""
util functions for optmlstat
"""

from typing import Any


def update_kwargs(kwargs: dict[str, Any], **default_kwargs) -> dict[str, Any]:
    default_kwargs.update(kwargs)
    return default_kwargs
