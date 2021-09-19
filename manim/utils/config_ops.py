"""Utilities that might be useful for configuration dictionaries."""

__all__ = [
    "merge_dicts_recursively",
    "update_dict_recursively",
    "DictAsObject",
]


import itertools as it
from dataclasses import dataclass

import numpy as np


def merge_dicts_recursively(*dicts):
    """
    Creates a dict whose keyset is the union of all the
    input dictionaries.  The value for each key is based
    on the first dict in the list with that key.

    dicts later in the list have higher priority

    When values are dictionaries, it is applied recursively
    """
    result = {}
    all_items = it.chain(*(d.items() for d in dicts))
    for key, value in all_items:
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_recursively(result[key], value)
        else:
            result[key] = value
    return result


def update_dict_recursively(current_dict, *others):
    updated_dict = merge_dicts_recursively(current_dict, *others)
    current_dict.update(updated_dict)


# Occasionally convenient in order to write dict.x instead of more laborious
# (and less in keeping with all other attr accesses) dict["x"]


class DictAsObject:
    def __init__(self, dictin):
        self.__dict__ = dictin


class _Data:
    """Descriptor that allows _Data variables to be grouped and accessed from self.data["attr"] via self.attr.
    self.data attributes must be arrays.
    """

    def __set_name__(self, obj, name):
        self.name = name

    def __get__(self, obj, owner):
        return obj.__dict__["data"][self.name]

    def __set__(self, obj, array: np.ndarray):
        obj.__dict__["data"][self.name] = array


class _Uniforms:
    """Descriptor that allows _Uniforms variables to be grouped from self.uniforms["attr"] via self.attr.
    self.uniforms attributes must be floats.
    """

    def __set_name__(self, obj, name):
        self.name = name

    def __get__(self, obj, owner):
        return obj.__dict__["uniforms"][self.name]

    def __set__(self, obj, num: float):
        obj.__dict__["uniforms"][self.name] = num
