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
    """
    flattened_parameters: list[T] = []
    for arg in args:
        # Mobject is iterable as it has `__iter__()`, but it should be appended.
        # To avoid cyclic import, we check for the `submobjects` attribute.
        if isinstance(arg, (Iterable, GeneratorType)) and not hasattr(arg, "submobjects"):
            flattened_parameters.extend(arg)
        else:
            flattened_parameters.append(arg)  # type: ignore[arg-type]
    return flattened_parameters
