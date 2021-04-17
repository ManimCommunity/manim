"""Utilities for expanding upon the standard :mod:`inspect` module."""

__all__ = ["is_classmethod", "new_methods_in_child_class"]

import inspect


def is_classmethod(obj):
    """Gets whether an object is a :class:`classmethod`.

    Parameters
    ----------
    obj
        The object to check.

    Returns
    -------
    :class:`bool`
        Whether the object is a :class:`classmethod`.
    """

    return inspect.ismethod(obj) and isinstance(obj.__self__, type)


def new_methods_in_child_class(child: type, parent: type):
    """Gets the new methods defined in a child class.

    Parameters
    ----------
    child
        The child class.
    parent
        The parent class.

    Returns
    -------
    :class:`typing.Iterable[function]`
        The newly defined methods.
    """

    # Local function to ease logic.
    # Returns the argument if it's a function,
    # returns the underlying function if it's
    # a classmethod, and otherwise returns None.
    def get_function(obj):
        if inspect.isfunction(obj):
            return obj

        if is_classmethod(obj):
            return obj.__func__

        return None

    for attr in dir(child):
        child_value = getattr(child, attr)
        child_value = get_function(child_value)
        if child_value is None:
            continue

        try:
            parent_value = getattr(parent, attr)
        except AttributeError:
            yield child_value
            continue

        parent_value = get_function(parent_value)
        if child_value is not parent_value:
            yield child_value
