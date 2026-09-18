"""Utilities that might be useful for configuration dictionaries."""

from __future__ import annotations

__all__ = [
    "merge_dicts_recursively",
    "update_dict_recursively",
    "DictAsObject",
]


import itertools as it
from typing import Any, Generic, Protocol, cast

import numpy.typing as npt
from typing_extensions import TypeVar


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


_Data_T = TypeVar("_Data_T", bound="npt.NDArray[Any]", default="npt.NDArray[Any]")


class _HasData(Protocol):
    data: dict[str, npt.NDArray[Any]]


class _Data(Generic[_Data_T]):
    """Descriptor that allows _Data variables to be grouped and accessed from self.data["attr"] via self.attr.
    self.data attributes must be arrays.
    """

    def __set_name__(self, obj: _HasData, name: str) -> None:
        self.name: str = name

    def __get__(self, obj: _HasData, owner: Any) -> _Data_T:
        value = cast(_Data_T, obj.data[self.name])
        return value

    def __set__(self, obj: _HasData, array: _Data_T) -> None:
        obj.data[self.name] = array


_Uniforms_T = TypeVar("_Uniforms_T", bound="float | tuple[float, ...]", default=float)


class _HasUniforms(Protocol):
    uniforms: dict[str, float | tuple[float, ...]]


class _Uniforms(Generic[_Uniforms_T]):
    """Descriptor that allows _Uniforms variables to be grouped from self.uniforms["attr"] via self.attr.
    self.uniforms attributes must be floats or tuples of floats.
    """

    def __set_name__(self, obj: _HasUniforms, name: str) -> None:
        self.name: str = name

    def __get__(self, obj: _HasUniforms, owner: Any) -> _Uniforms_T:
        val = cast(_Uniforms_T, obj.uniforms[self.name])
        return val

    def __set__(self, obj: _HasUniforms, num: _Uniforms_T) -> None:
        obj.uniforms[self.name] = num
