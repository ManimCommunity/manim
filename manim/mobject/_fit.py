"""Fit-to-size operations for :class:`~.Mobject`.

Standalone functions that take a mobject as their first argument; the
corresponding methods on :class:`Mobject` are thin delegations to these
helpers. This keeps :class:`Mobject`'s public API unchanged while moving
the implementation out of the already-large ``mobject.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from manim.mobject.mobject import Mobject


def rescale_to_fit(
    mob: Mobject, length: float, dim: int, stretch: bool = False, **kwargs: Any
) -> Mobject:
    old_length = mob.length_over_dim(dim)
    if old_length == 0:
        return mob
    if stretch:
        mob.stretch(length / old_length, dim, **kwargs)
    else:
        mob.scale(length / old_length, **kwargs)
    return mob


def scale_to_fit_width(mob: Mobject, width: float, **kwargs: Any) -> Mobject:
    return rescale_to_fit(mob, width, 0, stretch=False, **kwargs)


def stretch_to_fit_width(mob: Mobject, width: float, **kwargs: Any) -> Mobject:
    return rescale_to_fit(mob, width, 0, stretch=True, **kwargs)


def scale_to_fit_height(mob: Mobject, height: float, **kwargs: Any) -> Mobject:
    return rescale_to_fit(mob, height, 1, stretch=False, **kwargs)


def stretch_to_fit_height(mob: Mobject, height: float, **kwargs: Any) -> Mobject:
    return rescale_to_fit(mob, height, 1, stretch=True, **kwargs)


def scale_to_fit_depth(mob: Mobject, depth: float, **kwargs: Any) -> Mobject:
    return rescale_to_fit(mob, depth, 2, stretch=False, **kwargs)


def stretch_to_fit_depth(mob: Mobject, depth: float, **kwargs: Any) -> Mobject:
    return rescale_to_fit(mob, depth, 2, stretch=True, **kwargs)
