"""Utilities that might be useful for configuration dictionaries."""

from __future__ import annotations

__all__ = [
    "merge_dicts_recursively",
    "update_dict_recursively",
    "DictAsObject",
]


import itertools as it
from typing import Any


def merge_dicts_recursively(*dicts: dict[Any, Any]) -> dict[Any, Any]:
    """
    Creates a dict whose keyset is the union of all the
    input dictionaries.  The value for each key is based
    on the first dict in the list with that key.

    dicts later in the list have higher priority

    When values are dictionaries, it is applied recursively
    """
    result: dict = {}
    all_items = it.chain(*(d.items() for d in dicts))
    for key, value in all_items:
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_recursively(result[key], value)
        else:
            result[key] = value
    return result


def update_dict_recursively(
    current_dict: dict[Any, Any], *others: dict[Any, Any]
) -> None:
    updated_dict = merge_dicts_recursively(current_dict, *others)
    current_dict.update(updated_dict)


# Occasionally convenient in order to write dict.x instead of more laborious
# (and less in keeping with all other attr accesses) dict["x"]


class DictAsObject:
    def __init__(self, dictin: dict[str, Any]):
        self.__dict__ = dictin
