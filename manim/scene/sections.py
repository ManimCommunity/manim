from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Generic, ParamSpec, TypeVar, final, overload

from typing_extensions import TypedDict, Unpack

from manim.file_writer.sections import DefaultSectionType

__all__ = ["section"]


P = ParamSpec("P")
T = TypeVar("T")


class SceneSectionData(TypedDict, total=False):
    """(Public) data for a :class:`.SceneSection` in a :class:`.Scene`."""

    skip: bool
    type_: str
    name: str
    order: int


# mark as final because _cls_instance_count doesn't
# work with inheritance
@final
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

    def __init__(
        self, func: Callable[P, T], **kwargs: Unpack[SceneSectionData]
    ) -> None:
        self.func = func

        self.skip = False
        self.type_ = DefaultSectionType.NORMAL
        self.name = func.__name__

        # update the order for finding section orders
        self.order = self._cls_instance_count
        self.__class__._cls_instance_count += 1

        # we assume that users have a typechecker on
        # and aren't doing any weird stuff
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        name = self.name
        skip = self.skip
        section_type = self.type_
        order = self.order
        return f"{self.__class__.__name__}({name=}, {order=}, {skip=}, {section_type=})"

    def __repr__(self) -> str:
        # return a slightly more verbose repr
        s = str(self).removesuffix(")")
        func = self.func
        return f"{s}, {func=})"

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.func(*args, **kwargs)


@overload
def section(
    func: Callable[P, T],
    **kwargs: Unpack[SceneSectionData],
) -> SceneSection[P, T]: ...


@overload
def section(
    func: None = None,
    **kwargs: Unpack[SceneSectionData],
) -> Callable[[Callable[P, T]], SceneSection[P, T]]: ...


def section(
    func: Callable[P, T] | None = None, **kwargs: Unpack[SceneSectionData]
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
        return SceneSection(func, **kwargs)

    def wrapper(func: Callable[P, T]) -> SceneSection[P, T]:
        return SceneSection(func, **kwargs)

    return wrapper
