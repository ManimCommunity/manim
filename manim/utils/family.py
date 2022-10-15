from __future__ import annotations

import itertools as it
from typing import Iterable

from ..mobject.mobject import Mobject
from ..utils.iterables import remove_list_redundancies


def extract_mobject_family_members(
    mobjects: Iterable[Mobject],
    use_z_index=False,
    only_those_with_points: bool = False,
):
    """Returns a list of the types of mobjects and their family members present.
    A "family" in this context refers to a mobject, its submobjects, and their
    submobjects, recursively.

    Parameters
    ----------
    mobjects
        The Mobjects currently in the Scene
    only_those_with_points
        Whether or not to only do this for
        those mobjects that have points. By default False

    Returns
    -------
    list
        list of the mobjects and family members.
    """
    if only_those_with_points:
        method = Mobject.family_members_with_points
    else:
        method = Mobject.get_family
    extracted_mobjects = remove_list_redundancies(
        list(it.chain(*(method(m) for m in mobjects))),
    )
    if use_z_index:
        return sorted(extracted_mobjects, key=lambda m: m.z_index)
    return extracted_mobjects
