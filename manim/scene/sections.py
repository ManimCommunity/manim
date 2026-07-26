from __future__ import annotations

import types
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Generic, ParamSpec, Self, TypeVar, final

if TYPE_CHECKING:
    from .scene import Scene

__all__ = ["group"]


P = ParamSpec("P")
T = TypeVar("T")


# mark as final because _cls_instance_count doesn't
# work with inheritance
@final
class group(Generic[P, T]):
    """A group in a :class:`.Scene`.

    It holds data about each subsection, and keeps track of the order
    of the sections via :attr:`order`.

    Example
    -------

        .. code-block:: python

            class MyScene(Scene):
                groups_api = True

                @group
                def my_section(self):
                    pass

                @my_section
                def my_subsection(self):
                    pass

                @my_section
                def my_subsection2(self):
                    pass

    """

    _cls_instance_count: ClassVar[int] = 0
    """How many times the class has been instantiated.

    This is also used for ordering sections, because of the order
    decorators are called in a class.
    """

    def __init__(self, func: Callable[P, T]) -> None:
        self._func = func
        self._order = self.__class__._cls_instance_count

        self.__class__._cls_instance_count += 1

    @property
    def name(self) -> str:
        return self._func.__name__

    def __str__(self) -> str:
        name = self.name
        return f"{self.__class__.__name__}({name=})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._func!r})"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, group):
            return NotImplemented
        return self._order < other._order

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self._func(*args, **kwargs)

    def __get__(self, instance: Scene, _owner: type[Scene]) -> Self:
        """Descriptor to bind the group to the scene instance.

        This is called implicitly by python when methods are being bound.
        """
        self._func = types.MethodType(self._func, instance)
        return self
