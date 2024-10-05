"""Operations on iterables."""

from __future__ import annotations

__all__ = [
    "adjacent_n_tuples",
    "adjacent_pairs",
    "all_elements_are_instances",
    "concatenate_lists",
    "list_difference_update",
    "list_update",
    "listify",
    "make_even",
    "make_even_by_cycling",
    "remove_list_redundancies",
    "remove_nones",
    "stretch_array_to_length",
    "tuplify",
]

import itertools as it
from collections.abc import (
    Collection,
    Generator,
    Hashable,
    Iterable,
    Reversible,
    Sequence,
)
from typing import TYPE_CHECKING, Callable, TypeVar, overload

import numpy as np

T = TypeVar("T")
U = TypeVar("U")
F = TypeVar("F", np.float64, np.int_)
H = TypeVar("H", bound=Hashable)


if TYPE_CHECKING:
    import numpy.typing as npt


def adjacent_n_tuples(objects: Sequence[T], n: int) -> zip[tuple[T, ...]]:
    """Returns the Sequence objects cyclically split into n length tuples.

    See Also
    --------
    adjacent_pairs : alias with n=2

    Examples
    --------
    .. code-block:: pycon

        >>> list(adjacent_n_tuples([1, 2, 3, 4], 2))
        [(1, 2), (2, 3), (3, 4), (4, 1)]
        >>> list(adjacent_n_tuples([1, 2, 3, 4], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 1), (4, 1, 2)]
    """
    return zip(*([*objects[k:], *objects[:k]] for k in range(n)))


def adjacent_pairs(objects: Sequence[T]) -> zip[tuple[T, ...]]:
    """Alias for ``adjacent_n_tuples(objects, 2)``.

    See Also
    --------
    adjacent_n_tuples

    Examples
    --------
    .. code-block:: pycon

        >>> list(adjacent_pairs([1, 2, 3, 4]))
        [(1, 2), (2, 3), (3, 4), (4, 1)]
    """
    return adjacent_n_tuples(objects, 2)


def all_elements_are_instances(iterable: Iterable[object], Class: type[object]) -> bool:
    """Returns ``True`` if all elements of iterable are instances of Class.
    False otherwise.
    """
    return all(isinstance(e, Class) for e in iterable)


def batch_by_property(
    items: Iterable[T], property_func: Callable[[T], U]
) -> list[tuple[list[T], U | None]]:
    """Takes in a Sequence, and returns a list of tuples, (batch, prop)
    such that all items in a batch have the same output when
    put into the Callable property_func, and such that chaining all these
    batches together would give the original Sequence (i.e. order is
    preserved).

    Examples
    --------
    .. code-block:: pycon

        >>> batch_by_property([(1, 2), (3, 4), (5, 6, 7), (8, 9)], len)
        [([(1, 2), (3, 4)], 2), ([(5, 6, 7)], 3), ([(8, 9)], 2)]
    """
    batch_prop_pairs: list[tuple[list[T], U | None]] = []
    curr_batch: list[T] = []
    curr_prop = None
    for item in items:
        prop = property_func(item)
        if prop != curr_prop:
            # Add current batch
            if len(curr_batch) > 0:
                batch_prop_pairs.append((curr_batch, curr_prop))
            # Redefine curr
            curr_prop = prop
            curr_batch = [item]
        else:
            curr_batch.append(item)
    if len(curr_batch) > 0:
        batch_prop_pairs.append((curr_batch, curr_prop))
    return batch_prop_pairs


def concatenate_lists(*list_of_lists: Iterable[T]) -> list[T]:
    """Combines the Iterables provided as arguments into one list.

    Examples
    --------
    .. code-block:: pycon

        >>> concatenate_lists([1, 2], [3, 4], [5])
        [1, 2, 3, 4, 5]
    """
    return [item for lst in list_of_lists for item in lst]


def list_difference_update(l1: Iterable[T], l2: Iterable[T]) -> list[T]:
    """Returns a list containing all the elements of l1 not in l2.

    Examples
    --------
    .. code-block:: pycon

        >>> list_difference_update([1, 2, 3, 4], [2, 4])
        [1, 3]
    """
    return [e for e in l1 if e not in l2]


def list_update(l1: Iterable[T], l2: Iterable[T]) -> list[T]:
    """Used instead of ``set.update()`` to maintain order,
        making sure duplicates are removed from l1, not l2.
        Removes overlap of l1 and l2 and then concatenates l2 unchanged.

    Examples
    --------
    .. code-block:: pycon

        >>> list_update([1, 2, 3], [2, 4, 4])
        [1, 3, 2, 4, 4]
    """
    return [e for e in l1 if e not in l2] + list(l2)


@overload
def listify(obj: str) -> list[str]: ...


@overload
def listify(obj: Iterable[T]) -> list[T]: ...


@overload
def listify(obj: T) -> list[T]: ...


def listify(obj: str | Iterable[T] | T) -> list[str] | list[T]:
    """Converts obj to a list intelligently.

    Examples
    --------
    .. code-block:: pycon

        >>> listify("str")
        ['str']
        >>> listify((1, 2))
        [1, 2]
        >>> listify(len)
        [<built-in function len>]
    """
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, Iterable):
        return list(obj)
    else:
        return [obj]


def make_even(
    iterable_1: Iterable[T], iterable_2: Iterable[U]
) -> tuple[list[T], list[U]]:
    """Extends the shorter of the two iterables with duplicate values until its
        length is equal to the longer iterable (favours earlier elements).

    See Also
    --------
    make_even_by_cycling : cycles elements instead of favouring earlier ones

    Examples
    --------
    .. code-block:: pycon

        >>> make_even([1, 2], [3, 4, 5, 6])
        ([1, 1, 2, 2], [3, 4, 5, 6])

        >>> make_even([1, 2], [3, 4, 5, 6, 7])
        ([1, 1, 1, 2, 2], [3, 4, 5, 6, 7])
    """
    list_1, list_2 = list(iterable_1), list(iterable_2)
    len_list_1 = len(list_1)
    len_list_2 = len(list_2)
    length = max(len_list_1, len_list_2)
    return (
        [list_1[(n * len_list_1) // length] for n in range(length)],
        [list_2[(n * len_list_2) // length] for n in range(length)],
    )


def make_even_by_cycling(
    iterable_1: Collection[T], iterable_2: Collection[U]
) -> tuple[list[T], list[U]]:
    """Extends the shorter of the two iterables with duplicate values until its
        length is equal to the longer iterable (cycles over shorter iterable).

    See Also
    --------
    make_even : favours earlier elements instead of cycling them

    Examples
    --------
    .. code-block:: pycon

        >>> make_even_by_cycling([1, 2], [3, 4, 5, 6])
        ([1, 2, 1, 2], [3, 4, 5, 6])

        >>> make_even_by_cycling([1, 2], [3, 4, 5, 6, 7])
        ([1, 2, 1, 2, 1], [3, 4, 5, 6, 7])
    """
    length = max(len(iterable_1), len(iterable_2))
    cycle1 = it.cycle(iterable_1)
    cycle2 = it.cycle(iterable_2)
    return (
        [next(cycle1) for _ in range(length)],
        [next(cycle2) for _ in range(length)],
    )


def remove_list_redundancies(lst: Reversible[H]) -> list[H]:
    """Used instead of ``list(set(l))`` to maintain order.
    Keeps the last occurrence of each element.
    """
    reversed_result = []
    used = set()
    for x in reversed(lst):
        if x not in used:
            reversed_result.append(x)
            used.add(x)
    reversed_result.reverse()
    return reversed_result


def remove_nones(sequence: Iterable[T | None]) -> list[T]:
    """Removes elements where bool(x) evaluates to False.

    Examples
    --------
    .. code-block:: pycon

        >>> remove_nones(["m", "", "l", 0, 42, False, True])
        ['m', 'l', 42, True]
    """
    # Note this is redundant with it.chain
    return [x for x in sequence if x]


def resize_array(nparray: npt.NDArray[F], length: int) -> npt.NDArray[F]:
    """Extends/truncates nparray so that ``len(result) == length``.
        The elements of nparray are cycled to achieve the desired length.

    See Also
    --------
    resize_preserving_order : favours earlier elements instead of cycling them
    make_even_by_cycling : similar cycling behaviour for balancing 2 iterables

    Examples
    --------
    .. code-block:: pycon

        >>> points = np.array([[1, 2], [3, 4]])
        >>> resize_array(points, 1)
        array([[1, 2]])
        >>> resize_array(points, 3)
        array([[1, 2],
               [3, 4],
               [1, 2]])
        >>> resize_array(points, 2)
        array([[1, 2],
               [3, 4]])
    """
    if len(nparray) == length:
        return nparray
    return np.resize(nparray, (length, *nparray.shape[1:]))


def resize_preserving_order(
    nparray: npt.NDArray[np.float64], length: int
) -> npt.NDArray[np.float64]:
    """Extends/truncates nparray so that ``len(result) == length``.
        The elements of nparray are duplicated to achieve the desired length
        (favours earlier elements).

        Constructs a zeroes array of length if nparray is empty.

    See Also
    --------
    resize_array : cycles elements instead of favouring earlier ones
    make_even : similar earlier-favouring behaviour for balancing 2 iterables

    Examples
    --------
    .. code-block:: pycon

        >>> resize_preserving_order(np.array([]), 5)
        array([0., 0., 0., 0., 0.])

        >>> nparray = np.array([[1, 2], [3, 4]])
        >>> resize_preserving_order(nparray, 1)
        array([[1, 2]])

        >>> resize_preserving_order(nparray, 3)
        array([[1, 2],
               [1, 2],
               [3, 4]])
    """
    if len(nparray) == 0:
        return np.zeros((length, *nparray.shape[1:]))
    if len(nparray) == length:
        return nparray
    indices = np.arange(length) * len(nparray) // length
    return nparray[indices]


def resize_with_interpolation(nparray: npt.NDArray[F], length: int) -> npt.NDArray[F]:
    """Extends/truncates nparray so that ``len(result) == length``.
        New elements are interpolated to achieve the desired length.

        Note that if nparray's length changes, its dtype may too
        (e.g. int -> float: see Examples)

    See Also
    --------
    resize_array : cycles elements instead of interpolating
    resize_preserving_order : favours earlier elements instead of interpolating

    Examples
    --------
    .. code-block:: pycon

        >>> nparray = np.array([[1, 2], [3, 4]])
        >>> resize_with_interpolation(nparray, 1)
        array([[1., 2.]])
        >>> resize_with_interpolation(nparray, 4)
        array([[1.        , 2.        ],
               [1.66666667, 2.66666667],
               [2.33333333, 3.33333333],
               [3.        , 4.        ]])
        >>> nparray = np.array([[[1, 2], [3, 4]]])
        >>> nparray = np.array([[1, 2], [3, 4], [5, 6]])
        >>> resize_with_interpolation(nparray, 4)
        array([[1.        , 2.        ],
               [2.33333333, 3.33333333],
               [3.66666667, 4.66666667],
               [5.        , 6.        ]])
        >>> nparray = np.array([[1, 2], [3, 4], [1, 2]])
        >>> resize_with_interpolation(nparray, 4)
        array([[1.        , 2.        ],
               [2.33333333, 3.33333333],
               [2.33333333, 3.33333333],
               [1.        , 2.        ]])
    """
    if len(nparray) == length:
        return nparray
    cont_indices = np.linspace(0, len(nparray) - 1, length)
    return np.array(
        [
            (1 - a) * nparray[lh] + a * nparray[rh]
            for ci in cont_indices
            for lh, rh, a in [(int(ci), int(np.ceil(ci)), ci % 1)]
        ],
    )


def stretch_array_to_length(nparray: npt.NDArray[F], length: int) -> npt.NDArray[F]:
    # todo: is this the same as resize_preserving_order()?
    curr_len = len(nparray)
    if curr_len > length:
        raise Warning("Trying to stretch array to a length shorter than its own")
    indices = np.arange(length) / float(length)
    indices *= curr_len
    return nparray[indices.astype(int)]


@overload
def tuplify(obj: str) -> tuple[str]: ...


@overload
def tuplify(obj: Iterable[T]) -> tuple[T]: ...


@overload
def tuplify(obj: T) -> tuple[T]: ...


def tuplify(obj: str | Iterable[T] | T) -> tuple[str] | tuple[T]:
    """Converts obj to a tuple intelligently.

    Examples
    --------
    .. code-block:: pycon

        >>> tuplify("str")
        ('str',)
        >>> tuplify([1, 2])
        (1, 2)
        >>> tuplify(len)
        (<built-in function len>,)
    """
    if isinstance(obj, str):
        return (obj,)
    if isinstance(obj, Iterable):
        return tuple(obj)
    else:
        return (obj,)


def uniq_chain(*args: Iterable[T]) -> Generator[T, None, None]:
    """Returns a generator that yields all unique elements of the Iterables
        provided via args in the order provided.

    Examples
    --------
    .. code-block:: pycon

        >>> gen = uniq_chain([1, 2], [2, 3], [1, 4, 4])
        >>> from collections.abc import Generator
        >>> isinstance(gen, Generator)
        True
        >>> tuple(gen)
        (1, 2, 3, 4)
    """
    unique_items = set()
    for x in it.chain(*args):
        if x in unique_items:
            continue
        unique_items.add(x)
        yield x


def hash_obj(obj: object) -> int:
    """Determines a hash, even of potentially mutable objects."""
    if isinstance(obj, dict):
        return hash(tuple(sorted((hash_obj(k), hash_obj(v)) for k, v in obj.items())))

    if isinstance(obj, set):
        return hash(tuple(sorted(hash_obj(e) for e in obj)))

    if isinstance(obj, (tuple, list)):
        return hash(tuple(hash_obj(e) for e in obj))

    return hash(obj)
