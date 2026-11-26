"""Versioned attributes: a version that changes whenever the value is assigned.

A :class:`Field` is a class attribute that gives its value a version on every
assignment, so a reader can tell whether the value changed since it last looked
without comparing values. Mobjects declare the properties renderers draw from as
fields (``points``, colors, ``stroke_width``, ...), and renderers reuse their work
for a mobject while its versions stay the same.

The arrays of an :class:`ArrayField` are read-only: change them by assigning a new
array (``mobject.points = mobject.points + RIGHT``), not in place
(``mobject.points += RIGHT`` raises ``ValueError: ... read-only``). Since nothing can
write to a field's array, a version that did not change guarantees that the data did
not change either, and the array can be shared without a copy.

Versions come from one counter for all fields of all objects. An object's newest
assignment therefore always has its largest version, and the largest version of
several fields changes whenever any of them is assigned::

    square.points = ...  # points:       101
    square.fill_color = ...  # fill_color:   102
    circle.points = ...  # (circle's points: 103)
    square.stroke_width = 4  # stroke_width: 104; square's largest version: 104

A reader can summarize all fields of an object, or any group of them, by that one
number (see :attr:`.RNode.key`); with a counter per field, it would need all of them.
"""

from __future__ import annotations

import itertools as it
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, overload

import numpy as np
from typing_extensions import TypeIs

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

__all__ = [
    "ArrayField",
    "Field",
    "check_fields",
    "copy_versions",
    "field_version",
    "is_frozen",
]


T = TypeVar("T")

_NEXT_VERSION = it.count(1).__next__

_UNSET: Any = object()
# Types whose equal values are interchangeable and cheap to compare.
_EQ_TYPES = (int, float, bool, str)


class _Owner(np.ndarray):
    """Holds the data of a frozen field array; never handed out.

    A field stores a read-only plain array whose ``base`` is an ``_Owner``. Since the
    owner is read-only and nobody else has it, nothing can write to the data, and an
    array with such a base can be shared without a copy (by another field or by a
    copy of the object).
    """


def is_frozen(value: object) -> TypeIs[np.ndarray]:
    """Whether ``value`` is an array frozen by a field (see :class:`ArrayField`).

    Such an array cannot change, so it can be shared without a copy.
    """
    # A plain array only (not a subclass): views of other types are copied.
    return type(value) is np.ndarray and type(value.base) is _Owner  # type: ignore[attr-defined]


def _freeze(value: ArrayLike, dtype: DTypeLike | None) -> np.ndarray:
    """``value`` as a frozen array of ``dtype``: the value itself if it is frozen
    already, else a copy that nothing else can write to.
    """
    if is_frozen(value) and (dtype is None or value.dtype == dtype):
        return value
    array = np.asarray(value)
    owner = _Owner(array.shape, dtype=array.dtype if dtype is None else dtype)
    owner[...] = array
    owner.flags.writeable = False
    return owner.view(np.ndarray)


class Field(Generic[T]):
    """A versioned attribute.

    The value lives in the instance ``__dict__`` under the field's name, so code that
    walks ``__dict__`` (copying, hashing for the play cache) sees it as a plain
    attribute. Versions are kept per instance in ``_field_versions``; read them with
    :func:`field_version`.

    The descriptor only intercepts assignments; reads find the value in the instance
    ``__dict__`` directly and cost no more than for a plain attribute. A field must
    therefore be assigned before it is read (usually in ``__init__``): until then,
    reading it returns the descriptor itself.

    Values of a plain :class:`Field` are stored as given and must not be modified in
    place (numbers, enums, flags). Use :class:`ArrayField` for arrays.

    A subclass must not replace a field by a class attribute (``stroke_width = 8`` in
    the class body): assignments would bypass the field. :func:`check_fields` raises
    for that; :class:`~.Mobject` calls it for every subclass.
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

    def __set__(self, obj: object, value: T) -> None:
        _store(obj, self.name, value)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


class ArrayField(Field[T]):
    """A versioned array attribute.

    Assigned values are stored as read-only arrays of the field's dtype. The field
    copies what it is given, so no other reference can change the stored array, except
    for the value of a field, which nothing can change: it is stored as it is, and
    shared.

    Parameters
    ----------
    dtype
        The dtype assigned values are converted to; ``None`` keeps the dtype.
    """

    def __init__(self, dtype: DTypeLike | None = None) -> None:
        self.dtype = dtype

    def __set__(self, obj: object, value: ArrayLike) -> None:
        _store(obj, self.name, _freeze(value, self.dtype))


def _store(obj: object, name: str, value: object) -> None:
    attributes = obj.__dict__
    old = attributes.get(name, _UNSET)
    # Assigning the value the field holds, or an equal number, flag or string, is
    # not a change: keep the version. Arrays are compared by identity only.
    if old is value or (
        type(old) is type(value) and type(value) in _EQ_TYPES and old == value
    ):
        return
    attributes[name] = value
    version = _NEXT_VERSION()
    try:
        attributes["_field_versions"][name] = version
    except KeyError:
        attributes["_field_versions"] = {name: version}


def field_version(obj: object, name: str) -> int:
    """Return the version of the field ``name`` of ``obj``.

    The version changes with every assignment and is 0 while nothing has been
    assigned.
    """
    versions: dict[str, int] | None = obj.__dict__.get("_field_versions")
    return 0 if versions is None else versions.get(name, 0)


def copy_versions(source: object, target: object) -> None:
    """Give ``target`` the field versions of ``source``, as its own.

    For copies that take the field values of ``source`` directly (without assigning
    them): the copy then has the same versions, and assigning a field of one does not
    change the version of the other's.
    """
    versions = source.__dict__.get("_field_versions")
    if versions is not None:
        target.__dict__["_field_versions"] = versions.copy()


def check_fields(cls: type) -> None:
    """Raise if ``cls`` replaces a field of a base class by a plain class attribute.

    Assignments on instances would then bypass the field and get no version.

    Raises
    ------
    TypeError
        If a class attribute of ``cls`` hides a field of a base class.
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
