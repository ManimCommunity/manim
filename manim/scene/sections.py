from __future__ import annotations

import dataclasses
from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import ClassVar, Generic, ParamSpec, TypeVar, cast, final, overload

from manim.file_writer.sections import DefaultSectionType

__all__ = ["section"]


P = ParamSpec("P")
T = TypeVar("T")


# mark as final because _cls_instance_count doesn't
# work with inheritance
@final
@dataclasses.dataclass
class SceneSection(Generic[P, T]):
    """A section in a :class:`.Scene`.

    It holds data about each subsection, and keeps track of the order
    of the sections via :attr:`~SceneSection.order`.

    .. warning::

        :attr:`~SceneSection.func` is effectively a function - it is not
        bound to the scene, and thus must be called with the first argument
        as an instance of :class:`.Scene`.
    """

    _cls_instance_count: ClassVar[int] = 0
    """How many times the class has been instantiated.

    This is also used for ordering sections, because of the order
    decorators are called in a class.
    """

    func: Callable[P, T]
    _: dataclasses.KW_ONLY
    type_: str
    skip: bool = False
    override_name: str | None = None
    order: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.order = self._cls_instance_count
        # update the instance count
        self.__class__._cls_instance_count += 1

    @property
    def name(self) -> str:
        return self.func.__name__ if self.override_name is None else self.override_name

    def __str__(self) -> str:
        s = ""
        for field in dataclasses.fields(self):
            name = field.name
            if name == "func":
                s += f"name={self.name}"
            elif name == "override_name":
                continue
            else:
                attr = getattr(self, name)
                if isinstance(attr, Enum):
                    attr = attr.value
                s += f"{name}={attr}"
            s += ", "
        return f"{self.__class__.__name__}({s.removesuffix(', ')})"

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
                sections_api = True

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
