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
from typing import Any, Callable, Collection, Generator, Iterable, Reversible, Sequence

import numpy as np


def adjacent_n_tuples(objects: Sequence, n: int) -> zip:
    """Returns the Sequence objects cyclically split into n length tuples.

    See Also
    --------
    adjacent_pairs : alias with n=2

    Examples
    --------
    Normal usage::

        list(adjacent_n_tuples([1, 2, 3, 4], 2))
        # returns [(1, 2), (2, 3), (3, 4), (4, 1)]

        list(adjacent_n_tuples([1, 2, 3, 4], 3))
        # returns [(1, 2, 3), (2, 3, 4), (3, 4, 1), (4, 1, 2)]
    """
    return zip(*([*objects[k:], *objects[:k]] for k in range(n)))


def adjacent_pairs(objects: Sequence) -> zip:
    """Alias for ``adjacent_n_tuples(objects, 2)``.

    See Also
    --------
    adjacent_n_tuples

    Examples
    --------
    Normal usage::

        list(adjacent_pairs([1, 2, 3, 4]))
        # returns [(1, 2), (2, 3), (3, 4), (4, 1)]
    """
    return adjacent_n_tuples(objects, 2)


def all_elements_are_instances(iterable: Iterable, Class) -> bool:
    """Returns ``True`` if all elements of iterable are instances of Class.
    False otherwise.
    """
    return all([isinstance(e, Class) for e in iterable])


def batch_by_property(
    items: Sequence, property_func: Callable
) -> list[tuple[list, Any]]:
    """Takes in a Sequence, and returns a list of tuples, (batch, prop)
    such that all items in a batch have the same output when
    put into the Callable property_func, and such that chaining all these
    batches together would give the original Sequence (i.e. order is
    preserved).

    Examples
    --------
    Normal usage::

        batch_by_property([(1, 2), (3, 4), (5, 6, 7), (8, 9)], len)
        # returns [([(1, 2), (3, 4)], 2), ([(5, 6, 7)], 3), ([(8, 9)], 2)]
    """
    batch_prop_pairs = []
    curr_batch = []
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


def concatenate_lists(*list_of_lists: Iterable) -> list:
    """Combines the Iterables provided as arguments into one list.

    Examples
    --------
    Normal usage::

        concatenate_lists([1, 2], [3, 4], [5])
        # returns [1, 2, 3, 4, 5]
    """
    return [item for lst in list_of_lists for item in lst]


def list_difference_update(l1: Iterable, l2: Iterable) -> list:
    """Returns a list containing all the elements of l1 not in l2.

    Examples
    --------
    Normal usage::

        list_difference_update([1, 2, 3, 4], [2, 4])
        # returns [1, 3]
    """
    return [e for e in l1 if e not in l2]


def list_update(l1: Iterable, l2: Iterable) -> list:
    """Used instead of ``set.update()`` to maintain order,
        making sure duplicates are removed from l1, not l2.
        Removes overlap of l1 and l2 and then concatenates l2 unchanged.

    Examples
    --------
    Normal usage::

        list_update([1, 2, 3], [2, 4, 4])
        # returns [1, 3, 2, 4, 4]
    """
    return [e for e in l1 if e not in l2] + list(l2)


def listify(obj) -> list:
    """Converts obj to a list intelligently.

    Examples
    --------
    Normal usage::

        listify('str')   # ['str']
        listify((1, 2))  # [1, 2]
        listify(len)     # [<built-in function len>]
    """
    if isinstance(obj, str):
        return [obj]
    try:
        return list(obj)
    except TypeError:
        return [obj]


def make_even(iterable_1: Iterable, iterable_2: Iterable) -> tuple[list, list]:
    """Extends the shorter of the two iterables with duplicate values until its
        length is equal to the longer iterable (favours earlier elements).

    See Also
    --------
    make_even_by_cycling : cycles elements instead of favouring earlier ones

    Examples
    --------
    Normal usage::

        make_even([1, 2], [3, 4, 5, 6])
        ([1, 1, 2, 2], [3, 4, 5, 6])

        make_even([1, 2], [3, 4, 5, 6, 7])
        # ([1, 1, 1, 2, 2], [3, 4, 5, 6, 7])
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
    iterable_1: Collection, iterable_2: Collection
) -> tuple[list, list]:
    """Extends the shorter of the two iterables with duplicate values until its
        length is equal to the longer iterable (cycles over shorter iterable).

    See Also
    --------
    make_even : favours earlier elements instead of cycling them

    Examples
    --------
    Normal usage::

        make_even_by_cycling([1, 2], [3, 4, 5, 6])
        ([1, 2, 1, 2], [3, 4, 5, 6])

        make_even_by_cycling([1, 2], [3, 4, 5, 6, 7])
        # ([1, 2, 1, 2, 1], [3, 4, 5, 6, 7])
    """
    length = max(len(iterable_1), len(iterable_2))
    cycle1 = it.cycle(iterable_1)
    cycle2 = it.cycle(iterable_2)
    return (
        [next(cycle1) for _ in range(length)],
        [next(cycle2) for _ in range(length)],
    )


def remove_list_redundancies(lst: Reversible) -> list:
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


def remove_nones(sequence: Iterable) -> list:
    """Removes elements where bool(x) evaluates to False.

    Examples
    --------
    Normal usage::

        remove_nones(['m', '', 'l', 0, 42, False, True])
        # ['m', 'l', 42, True]
    """
    # Note this is redundant with it.chain
    return [x for x in sequence if x]


def resize_array(nparray: np.ndarray, length: int) -> np.ndarray:
    """Extends/truncates nparray so that ``len(result) == length``.
        The elements of nparray are cycled to achieve the desired length.

    See Also
    --------
    resize_preserving_order : favours earlier elements instead of cycling them
    make_even_by_cycling : similar cycling behaviour for balancing 2 iterables

    Examples
    --------
    Normal usage::

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


def resize_preserving_order(nparray: np.ndarray, length: int) -> np.ndarray:
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
    Normal usage::

        resize_preserving_order(np.array([]), 5)
        # np.array([0., 0., 0., 0., 0.])

        nparray = np.array([[1, 2],
                            [3, 4]])

        resize_preserving_order(nparray, 1)
        # np.array([[1, 2]])

        resize_preserving_order(nparray, 3)
        # np.array([[1, 2],
        #           [1, 2],
        #           [3, 4]])
    """
    if len(nparray) == 0:
        return np.zeros((length, *nparray.shape[1:]))
    if len(nparray) == length:
        return nparray
    indices = np.arange(length) * len(nparray) // length
    return nparray[indices]


def resize_with_interpolation(nparray: np.ndarray, length: int) -> np.ndarray:
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
    Normal usage::

        nparray = np.array([[1, 2],
                            [3, 4]])

        resize_with_interpolation(nparray, 1)
        # np.array([[1., 2.]])

        resize_with_interpolation(nparray, 4)
        # np.array([[1.        , 2.        ],
        #           [1.66666667, 2.66666667],
        #           [2.33333333, 3.33333333],
        #           [3.        , 4.        ]])

        nparray = np.array([[[1, 2],[3, 4]]])
        resize_with_interpolation(nparray, 3)
        # np.array([[[1., 2.], [3., 4.]],
        #           [[1., 2.], [3., 4.]],
        #           [[1., 2.], [3., 4.]]])

        nparray = np.array([[1, 2], [3, 4], [5, 6]])
        resize_with_interpolation(nparray, 4)
        # np.array([[1.        , 2.        ],
        #           [2.33333333, 3.33333333],
        #           [3.66666667, 4.66666667],
        #           [5.        , 6.        ]])

        nparray = np.array([[1, 2], [3, 4], [1, 2]])
        resize_with_interpolation(nparray, 4)
        # np.array([[1.        , 2.        ],
        #           [2.33333333, 3.33333333],
        #           [2.33333333, 3.33333333],
        #           [1.        , 2.        ]])
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


def stretch_array_to_length(nparray: np.ndarray, length: int) -> np.ndarray:
    # todo: is this the same as resize_preserving_order()?
    curr_len = len(nparray)
    if curr_len > length:
        raise Warning("Trying to stretch array to a length shorter than its own")
    indices = np.arange(length) / float(length)
    indices *= curr_len
    return nparray[indices.astype(int)]


def tuplify(obj) -> tuple:
    """Converts obj to a tuple intelligently.

    Examples
    --------
    Normal usage::

        tuplify('str')   # ('str',)
        tuplify([1, 2])  # (1, 2)
        tuplify(len)     # (<built-in function len>,)
    """
    if isinstance(obj, str):
        return (obj,)
    try:
        return tuple(obj)
    except TypeError:
        return (obj,)


def uniq_chain(*args: Iterable) -> Generator:
    """Returns a generator that yields all unique elements of the Iterables
        provided via args in the order provided.

    Examples
    --------
    Normal usage::

        uniq_chain([1, 2], [2, 3], [1, 4, 4])
        # yields 1, 2, 3, 4
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
