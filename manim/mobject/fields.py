"""Versioned mobject properties.

A :class:`Field` is a mobject property that renderers draw from. Assigning to it
changes the field's version, so a renderer can tell whether the value changed since it
last looked.

The arrays of an :class:`ArrayField` are read-only: change them by assigning a new
array (``mobject.points = mobject.points + RIGHT``), not in place
(``mobject.points += RIGHT`` raises ``ValueError: ... read-only``).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, overload

import numpy as np

__all__ = ["ArrayField", "Field", "field_version"]


T = TypeVar("T")

# Versions come from one counter: a new value always gets a version that no value of
# this field had before, which is all a reader needs.
_next_version = itertools.count(1).__next__


class _Owner(np.ndarray):
    """Holds the data of a frozen field array; never handed out.

    A field stores a read-only plain array whose ``base`` is an ``_Owner``. Since the
    owner is read-only and nobody else has it, nothing can write to the data, and an
    array with such a base can be shared without a copy: by another field, or by a
    copy of the mobject.
    """


def _is_frozen(value: object) -> bool:
    """Whether ``value`` is an array frozen by a field (see :class:`_Owner`)."""
    # A plain array only: views of other types are copied.
    return type(value) is np.ndarray and type(value.base) is _Owner  # type: ignore[attr-defined]


def _freeze(value: Any, dtype: type[np.generic] | None) -> np.ndarray:
    """``value`` as a frozen array of ``dtype``: the value itself if it is frozen
    already, else a copy that nothing else can write to.
    """
    if _is_frozen(value) and (dtype is None or value.dtype == dtype):
        return value
    value = np.asarray(value)
    owner = _Owner(value.shape, dtype=dtype or value.dtype)
    owner[...] = value
    owner.flags.writeable = False
    return owner.view(np.ndarray)


class Field(Generic[T]):
    """A versioned property of a mobject.

    The value lives in the instance ``__dict__`` under the field's name, so code that
    walks ``__dict__`` (copying, hashing for the play cache) sees it as a plain
    attribute. Versions are kept per instance in ``_field_versions``.

    The descriptor only intercepts assignments; reads find the value in the instance
    ``__dict__`` directly and cost no more than for a plain attribute. A field must
    therefore be assigned before it is read (usually in ``__init__``): until then,
    reading it returns the descriptor itself.

    Values of a plain :class:`Field` are stored as given and must not be modified in
    place (numbers, enums, flags). Use :class:`ArrayField` for arrays.

    A subclass must not replace a field by a class attribute (``stroke_width = 8`` in
    the class body); set the value in ``__init__`` instead.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    if TYPE_CHECKING:

        @overload
        def __get__(self, obj: None, owner: type | None = None) -> Self: ...

        @overload
        def __get__(self, obj: object, owner: type | None = None) -> T: ...

        def __get__(
            self, obj: object | None, owner: type | None = None
        ) -> Self | T: ...

    def __set__(self, obj: object, value: Any) -> None:
        _store(obj, self.name, value)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


class ArrayField(Field[T]):
    """A versioned array property of a mobject.

    Assigned values are stored as read-only arrays of the field's dtype. The field
    copies what it is given, so no other reference can change the stored array, except
    for the value of a field, which nothing can change: it is stored as it is, and
    shared.

    Parameters
    ----------
    dtype
        The dtype assigned values are converted to; ``None`` keeps the dtype.
    """

    def __init__(self, dtype: type[np.generic] | None = None) -> None:
        self.dtype = dtype

    def __set__(self, obj: object, value: Any) -> None:
        _store(obj, self.name, _freeze(value, self.dtype))


def _store(obj: object, name: str, value: Any) -> None:
    attributes = obj.__dict__
    attributes[name] = value
    try:
        attributes["_field_versions"][name] = _next_version()
    except KeyError:
        attributes["_field_versions"] = {name: _next_version()}


def field_version(obj: object, name: str) -> int:
    """Return the version of the field ``name`` of ``obj``.

    The version changes with every assignment and is 0 while nothing has been
    assigned.
    """
    version: int = obj.__dict__.get("_field_versions", {}).get(name, 0)
    return version


def _check_no_replaced_fields(cls: type) -> None:
    """Raise if ``cls`` replaces a field of a base class by a plain class attribute.

    Assignments on instances would then bypass the field and get no version.
    """
    for name, value in vars(cls).items():
        if isinstance(value, Field):
            continue
        for base in cls.__mro__[1:]:
            if name in vars(base):
                if isinstance(vars(base)[name], Field):
                    raise TypeError(
                        f"{cls.__qualname__}.{name} replaces the field "
                        f"{base.__qualname__}.{name}: set the value in __init__ "
                        "instead of in the class body."
                    )
                break
