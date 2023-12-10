from __future__ import annotations

from types import GeneratorType
from typing import Iterable, TypeVar

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
