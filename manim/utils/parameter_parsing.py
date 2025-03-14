from __future__ import annotations

from collections.abc import Iterable
from types import GeneratorType
from typing import TypeVar

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

    Notes
    -----
    Instances of :class:`Mobject` are technically iterable because they define
    `__iter__()`, but they should be treated as single objects rather than
    being expanded. To prevent unintended behavior, we explicitly check
    `not isinstance(arg, Mobject)` before extending the list.
    """
    # avoiding cyclic import
    from ..mobject.mobject import Mobject

    flattened_parameters: list[T] = []
    for arg in args:
        # Only extend if arg is iterable and NOT an instance of Mobject
        if isinstance(arg, (Iterable, GeneratorType)) and not isinstance(arg, Mobject):
            flattened_parameters.extend(arg)
        else:
            flattened_parameters.append(arg)
    return flattened_parameters
