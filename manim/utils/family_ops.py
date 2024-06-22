from __future__ import annotations

import itertools as it

__all__ = [
    "extract_mobject_family_members",
    "restructure_list_to_exclude_certain_family_members",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from manim.mobject.mobject import Mobject


def extract_mobject_family_members(
    mobject_list: Iterable[Mobject], only_those_with_points: bool = False
) -> Sequence[Mobject]:
    result = list(it.chain(*(mob.get_family() for mob in mobject_list)))
    if only_those_with_points:
        result = [mob for mob in result if mob.has_points()]
    return result


def restructure_list_to_exclude_certain_family_members(
    mobject_list: Iterable[Mobject], to_remove: Iterable[Mobject]
) -> Sequence[Mobject]:
    """
    Removes anything in to_remove from mobject_list, but in the event that one of
    the items to be removed is a member of the family of an item in mobject_list,
    the other family members are added back into the list.

    This is useful in cases where a scene contains a group, e.g. Group(m1, m2, m3),
    but one of its submobjects is removed, e.g. scene.remove(m1), it's useful
    for the list of mobject_list to be edited to contain other submobjects, but not m1.
    """
    new_list = []
    to_remove = extract_mobject_family_members(to_remove)

    def add_safe_mobjects_from_list(list_to_examine, set_to_remove):
        for mob in list_to_examine:
            if mob in set_to_remove:
                continue
            intersect = set_to_remove.intersection(mob.get_family())
            if intersect:
                add_safe_mobjects_from_list(mob.submobjects, intersect)
            else:
                new_list.append(mob)

    add_safe_mobjects_from_list(mobject_list, set(to_remove))
    return new_list


def recursive_mobject_remove(
    mobjects: list[Mobject], to_remove: set[Mobject]
) -> tuple[Sequence[Mobject], bool]:
    """
    Takes in a list of mobjects, together with a set of mobjects to remove.
    The first component of what's removed is a new list such that any mobject
    with one of the elements from `to_remove` in its family is no longer in
    the list, and in its place are its family members which aren't in `to_remove`
    The second component is a boolean value indicating whether any removals were made
    """
    result = []
    found_in_list = False
    for mob in mobjects:
        if mob in to_remove:
            found_in_list = True
            continue
        # Recursive call
        sub_list, found_in_submobjects = recursive_mobject_remove(
            mob.submobjects, to_remove
        )
        if found_in_submobjects:
            result.extend(sub_list)
            found_in_list = True
        else:
            result.append(mob)
    return result, found_in_list
