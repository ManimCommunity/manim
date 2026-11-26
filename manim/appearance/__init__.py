"""How mobjects look: the paints of their areas and lines, their materials, and
the lights of a scene.

The names listed here can be imported from :mod:`manim.appearance` directly (and
from ``manim``).

Modules
=======

.. autosummary::
    :toctree: ../reference

    ~light
    ~material
    ~paint
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .material import Material, PbrMaterial, UnlitMaterial, interpolate_material
from .paint import ColorField, LinearGradient, Paint, SolidColor, interpolate_paint

if TYPE_CHECKING:
    from .light import (
        DEFAULT_AMBIENT_LIGHT,
        DEFAULT_LIGHT_INTENSITY,
        DEFAULT_LIGHT_POSITION,
        PointLight,
    )

# PointLight is a mobject, and manim.mobject imports this package (for Paint):
# the names of the light module are imported when they are first used.
_LIGHT_NAMES = frozenset(
    {
        "DEFAULT_AMBIENT_LIGHT",
        "DEFAULT_LIGHT_INTENSITY",
        "DEFAULT_LIGHT_POSITION",
        "PointLight",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LIGHT_NAMES:
        from . import light

        return getattr(light, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_AMBIENT_LIGHT",
    "DEFAULT_LIGHT_INTENSITY",
    "DEFAULT_LIGHT_POSITION",
    "ColorField",
    "LinearGradient",
    "Material",
    "Paint",
    "PbrMaterial",
    "PointLight",
    "SolidColor",
    "UnlitMaterial",
    "interpolate_material",
    "interpolate_paint",
]
