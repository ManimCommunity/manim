from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import MethodType
from typing import Any


@dataclass
class MethodWithArgs:
    __slots__ = ["method", "args", "kwargs"]
    method: MethodType
    args: Iterable[Any]
    kwargs: dict[str, Any]


class SceneInteractRerun:
    __slots__ = ["sender", "kwargs"]

    def __init__(self, sender: str, **kwargs: Any) -> None:
        self.sender = sender
        self.kwargs = kwargs


class SceneInteractExit:
    __slots__ = ["sender"]

    def __init__(self, sender: str) -> None:
        self.sender = sender