from __future__ import annotations

import types
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Generic, ParamSpec, TypeVar, final, overload

from typing_extensions import Self, TypedDict, Unpack

if TYPE_CHECKING:
    from .scene import Scene

__all__ = ["group"]


P = ParamSpec("P")
T = TypeVar("T")


class SectionGroupData(TypedDict, total=False):
    """(Public) data for a :class:`.SectionGroup` in a :class:`.Scene`."""

    skip: bool
    order: int


# mark as final because _cls_instance_count doesn't
# work with inheritance
@final
class SectionGroup(Generic[P, T]):
    """A section in a :class:`.Scene`.

    It holds data about each subsection, and keeps track of the order
    of the sections via :attr:`~SectionGroup.order`.

    .. warning::

        :attr:`~SectionGroup.func` is effectively a function - it is not
        bound to the scene, and thus must be called with the first argument
        as an instance of :class:`.Scene`.
    """

    _cls_instance_count: ClassVar[int] = 0
    """How many times the class has been instantiated.

    This is also used for ordering sections, because of the order
    decorators are called in a class.
    """

    def __init__(
        self, func: Callable[P, T], **kwargs: Unpack[SectionGroupData]
    ) -> None:
        self.func = func

        self.skip = kwargs.get("skip", False)

        # update the order counter
        self.order = self._cls_instance_count
        self.__class__._cls_instance_count += 1
        if "order" in kwargs:
            self.order = kwargs["order"]

    def __str__(self) -> str:
        skip = self.skip
        order = self.order
        return f"{self.__class__.__name__}({order=}, {skip=})"

    def __repr__(self) -> str:
        # return a slightly more verbose repr
        s = str(self).removesuffix(")")
        func = self.func
        return f"{s}, {func=})"

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.func(*args, **kwargs)

    def bind(self, instance: Scene) -> Self:
        """Binds :attr:`func` to the scene instance, making :attr:`func` a method.

        This allows the section to be called without the scene being passed explicitly.
        """
        self.func = types.MethodType(self.func, instance)
        return self

    def __get__(self, instance: Scene, _owner: type[Scene]) -> Self:
        """Descriptor to bind the section to the scene instance.

        This is called implicitly by python when methods are being bound.
        """
        return self  # HELPME use binding
        return self.bind(instance)


@overload
def group(
    func: Callable[P, T],
    **kwargs: Unpack[SectionGroupData],
) -> SectionGroup[P, T]: ...


@overload
def group(
    func: None = None,
    **kwargs: Unpack[SectionGroupData],
) -> Callable[[Callable[P, T]], SectionGroup[P, T]]: ...


def group(
    func: Callable[P, T] | None = None, **kwargs: Unpack[SectionGroupData]
) -> SectionGroup[P, T] | Callable[[Callable[P, T]], SectionGroup[P, T]]:
    r"""Decorator to create a SectionGroup in the scene.

    Example
    -------

        .. code-block:: python

            class MyScene(Scene):
                SectionGroups_api = True

                @SectionGroup
                def first_SectionGroup(self):
                    pass

                @SectionGroup(skip=True)
                def second_SectionGroup(self):
                    pass

    Parameters
    ----------
        func : Callable
            The subsection.
        skip : bool, optional
            Whether to skip the section, by default False
    """

    def wrapper(func: Callable[P, T]) -> SectionGroup[P, T]:
        return SectionGroup(func, **kwargs)

    return wrapper(func) if func is not None else wrapper
