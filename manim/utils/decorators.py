from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def internal(f: F, /) -> F:
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
    # we have to keep the "No description provided" or
    # docutils doesn't parse the directive properly
    doc = f.__doc__ or "No description provided"

    directive = (
        ".. warning::\n\n    "
        "This method is designed for internal use and may not stay the same in future versions of Manim. "
        "Use these in your code at your own risk"
    )

    f.__doc__ = f"{directive}\n\n{inspect.cleandoc(doc)}"
    return f
