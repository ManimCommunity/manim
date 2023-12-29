from __future__ import annotations

from types import GeneratorType
from typing import Iterable, TypeVar, Any

T = TypeVar("T")


def flatten_iterable_parameters(
    args: Iterable[T | Iterable[T] | GeneratorType],
) -> list[T]:
    """Flattens an iterable of parameters into a list of parameters.

    Parameters
    ----------
    args
        The iterable of parameters to flatten.
        [(generator), [], (), ...]

    Returns
    -------
    :class:`list`
        The flattened list of parameters.
    """
    flattened_parameters = []
    for arg in args:
        if isinstance(arg, (Iterable, GeneratorType)):
            flattened_parameters.extend(arg)
        else:
            flattened_parameters.append(arg)
    return flattened_parameters


def flatten_iterable_parameters_excluding_cls(
    args: Iterable[T | Iterable[T] | GeneratorType],
    classes: tuple[type],
    bad_classes: tuple[type[Any]] = (),
    error_message: str = "Invalid class %(arg)s"
) -> list[T]:
    """Flattens an iterable of parameters into a list of parameters, giving priority to
    adding certain classes to the flattened list first.

    Parameters
    ----------
    args
        The Iterable of parameters to flatten.
        [(generator), [], (), ...]
    classes
        If ``type(arg) in classes`` the arg will be
        added to the flattened list without any further
        processing
    """
    flattened = []
    for arg in args:
        if isinstance(arg, classes):
            flattened.append(arg)
        elif isinstance(arg, bad_classes):
            raise TypeError(error_message % {"arg": arg})
        elif isinstance(arg, (Iterable, GeneratorType)):
            # TODO:
            # Figure out a way to check if there are any
            # ``bad_classes`` in arg before extending list
            # Recursion won't work for large VGroups
            flattened.extend(arg)
        else:
            flattened.append(arg)
    return flattened
