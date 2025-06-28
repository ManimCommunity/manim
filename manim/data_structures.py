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


@dataclass
class SceneInteractContinue:
    """Object which, when encountered in :meth:`~.Scene.interact`, triggers
    the end of the scene interaction, continuing with the rest of the
    animations, if any.

    Attributes
    ----------
    sender : str
        The name of the entity which issued the end of the scene interaction,
        such as "gui" or "keyboard".
    """

    __slots__ = ["sender"]

    sender: str


class SceneInteractRerun:
    """Object which, when encountered in :meth:`~.Scene.interact`, triggers
    the rerun of the scene.

    Attributes
    ----------
    sender : str
        The name of the entity which issued the rerun of the scene, such as
        "gui", "keyboard", "play" or "file".
    kwargs : dict[str, Any]
        Additional keyword arguments when rerunning the scene. Currently,
        only `"from_animation_number"` is being used, which determines the
        animation from which to start rerunning the scene.
    """

    __slots__ = ["sender", "kwargs"]

    def __init__(self, sender: str, **kwargs: Any) -> None:
        self.sender = sender
        self.kwargs = kwargs
