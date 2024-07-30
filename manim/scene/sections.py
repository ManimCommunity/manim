from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Generic, ParamSpec, TypeVar, cast, overload

from manim.file_writer.sections import DefaultSectionType

__all__ = ["section"]


P = ParamSpec("P")
T = TypeVar("T")


class SceneSection(Generic[P, T]):
    """A section in a :class:`.Scene`.

    It holds data about each subsection, and keeps track of the order
    of the sections via :attr:`~SceneSection.order`.

    .. warning::

        :attr:`~SceneSection.func` is effectively a function - it is not
        bound to the scene, and thus must be called with the first argument
        as an instance of :class:`.Scene`.
    """

    _cls_instance_count = 0
    """How many times the class has been instantiated.

    This is also used for ordering sections, because of the order
    decorators are called in a class.
    """

    def __init__(
        self,
        func: Callable[P, T],
        *,
        type_: str,
        skip: bool = False,
        override_name: str | None = None,
    ) -> None:
        self.func = func
        self.order = self._cls_instance_count
        self.skip = skip
        self.type_ = type_
        self._override_name = override_name

        # update the instance count
        self.__class__._cls_instance_count += 1

    @property
    def name(self) -> str:
        return (
            self.func.__name__ if self._override_name is None else self._override_name
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name}, {self.order}, {self.skip})"

    __repr__ = __str__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.func(*args, **kwargs)


@overload
def section(
    func: Callable[P, T],
    *,
    skip: bool = False,
    type_: str = DefaultSectionType.NORMAL,
    name: str | None = None,
) -> SceneSection[P, T]: ...


@overload
def section(
    func: None = None,
    *,
    skip: bool = False,
    type_: str = DefaultSectionType.NORMAL,
    name: str | None = None,
) -> Callable[[Callable[P, T]], SceneSection[P, T]]: ...


def section(
    func: Callable[P, T] | None = None,
    *,
    skip: bool = False,
    type_: str = DefaultSectionType.NORMAL,
    name: str | None = None,
) -> SceneSection[P, T] | Callable[[Callable[P, T]], SceneSection[P, T]]:
    r"""Decorator to create a section in the scene.

    Example
    -------

        .. code-block:: python

            class MyScene(Scene):
                use_sections_api = True

                @section
                def first_section(self):
                    pass

                @section(skip=True, name="Introduce Bob")
                def second_section(self):
                    pass

    Parameters
    ----------
        func : Callable
            The subsection.
        skip : bool, optional
            Whether to skip the section, by default False
        type\_ : str, optional
            The type of the section, by default :attr:`.DefaultSectionType.NORMAL`
        name : str, optional
            The name of the section, by default the name of the method.
    """
    if func is not None:
        return SceneSection(
            func,
            type_=type_,
            skip=skip,
            override_name=name,
        )
    return cast(Callable, partial(section, skip=skip, type_=type_))
