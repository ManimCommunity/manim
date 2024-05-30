from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import ParamSpec

    P = ParamSpec("P")
    T = TypeVar("T")


def internal(f: Callable[P, T]) -> Callable[P, T]:
    """
    This decorator marks a function as internal
    by adding a warning to the docstring of the object.

    Note that usage on a method starting with an underscore
    has no effect, as sphinx does not document such methods

    .. code-block:: python

        @internal
        def some_private_method(self):
            # does some private stuff
            ...


        @internal  # does not do anything, don't use
        def _my_second_private_method(self): ...
    """
    doc: str = f.__doc__ if f.__doc__ is not None else ""
    newblockline = "\n    "
    directive = f".. warning::{newblockline}"
    directive += newblockline.join(
        (
            "This method is designed for internal use",
            "and may not stay the same in future versions of Manim",
        )
    )
    f.__doc__ = f"{directive}\n\n{doc}"
    return f
