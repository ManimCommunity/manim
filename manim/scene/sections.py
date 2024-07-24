from __future__ import annotations

from collections.abc import Callable
from functools import partialmethod
from typing import Generic, ParamSpec, TypeVar, cast, overload

from manim.file_writer.sections import DefaultSectionType

__all__ = ["section"]


P = ParamSpec("P")
T = TypeVar("T")


class SceneSection(Generic[P, T]):
    def __init__(
        self, func: Callable[P, T], *, order: int, type_: str, skip: bool = False
    ) -> None:
        self.func = func
        self.order = order
        self.skip = skip
        self.type_ = type_

    @property
    def name(self) -> str:
        return self.func.__name__

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name}, {self.order}, {self.skip})"

    __repr__ = __str__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.func(*args, **kwargs)


@overload
def section(
    func: Callable[P, T], *, skip: bool = False, type_: str | None = None
) -> SceneSection[P, T]: ...


@overload
def section(
    func: None = None, *, skip: bool = False, type_: str | None = None
) -> Callable[[Callable[P, T]], SceneSection[P, T]]: ...


def section(
    func: Callable[P, T] | None = None,
    *,
    skip: bool = False,
    type_: str | None = None,
) -> SceneSection[P, T] | Callable[[Callable[P, T]], SceneSection[P, T]]:
    """Decorator to create a section in the scene."""
    if func is not None:
        section._section_order += 1  # type: ignore
        return SceneSection(
            func,
            order=section._section_order,  # type: ignore
            type_=type_ or DefaultSectionType.NORMAL,
        )

    return cast(Callable, partialmethod(section, skip=skip, type_=type_))


section._section_order = 0  # type: ignore
