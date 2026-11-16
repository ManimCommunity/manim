"""Data classes and other necessary data structures for use in Manim."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import MethodType
from typing import Any


@dataclass
class MethodWithArgs:
    """Object containing a :attr:`method` which is intended to be called later
    with the positional arguments :attr:`args` and the keyword arguments
    :attr:`kwargs`.

    Attributes
    ----------
    method : MethodType
        A callable representing a method of some class.
    args : Iterable[Any]
        Positional arguments for :attr:`method`.
    kwargs : dict[str, Any]
        Keyword arguments for :attr:`method`.
    """

    __slots__ = ["method", "args", "kwargs"]

    method: MethodType
    args: Iterable[Any]
    kwargs: dict[str, Any]
