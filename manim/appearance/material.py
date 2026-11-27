"""How a surface responds to light.

A mobject's material says whether and how it is lit. Its color comes from the mobject
(its fill or stroke); the material adds how light falls on it. Materials are immutable
values: compare them with ``==``, and assign a new one to change a mobject's material
(:meth:`.VMobject.set_material`, :meth:`.MeshMobject.set_material`).

Meshes, surfaces and :class:`.ThreeDVMobject` are :class:`PbrMaterial` by default;
every other mobject is :class:`UnlitMaterial`. Mobjects fixed in the frame are never
lit, and without lights nothing is shaded. See :doc:`/guides/colors_materials_lights`.

Materials of your own subclass :class:`Material`: a frozen dataclass whose fields are
its parameters, overriding :meth:`Material.interpolate` to animate them. Renderers
draw the materials they know and raise :class:`~.CapabilityError` for any other, so a
new material also needs a renderer that draws it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Material", "PbrMaterial", "UnlitMaterial", "interpolate_material"]


@dataclass(frozen=True, slots=True)
class Material:
    """How a surface responds to light: the base class of materials.

    A material is an immutable value (subclasses are frozen dataclasses, so that
    they compare, hash and copy by their parameters). :class:`UnlitMaterial` and
    :class:`PbrMaterial` are the built-in ones.
    """

    def interpolate(self, end: Material, alpha: float) -> Material:
        """The material at progress ``alpha`` (0 to 1) from this one to ``end``.

        By default it switches to ``end`` when ``alpha`` reaches 1, so that a
        transform between materials that cannot be blended (a lit and an unlit one,
        say) still works; override it to animate a material's parameters.
        """
        return end if alpha >= 1 else self


@dataclass(frozen=True, slots=True)
class UnlitMaterial(Material):
    """Not lit: drawn in its own color, whatever lights there are."""


@dataclass(frozen=True, slots=True)
class PbrMaterial(Material):
    """Lit as a physically based surface (glTF 2.0 metallic-roughness).

    Parameters
    ----------
    roughness
        From 0 (smooth: sharp highlights) to 1 (rough: no highlights).
    metallic
        From 0 (a dielectric, like plastic or paint) to 1 (a metal, whose reflections
        take its color).
    emissive
        Light the surface emits itself, as linear RGB components that are not
        negative; black by default.
    """

    roughness: float = 0.5
    metallic: float = 0.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("roughness", "metallic"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        emissive = tuple(float(c) for c in self.emissive)
        if len(emissive) != 3 or not all(math.isfinite(c) and c >= 0 for c in emissive):
            raise ValueError(
                "emissive must be three finite components that are not negative, "
                f"got {self.emissive}"
            )
        object.__setattr__(self, "emissive", emissive)

    def interpolate(self, end: Material, alpha: float) -> Material:
        """Towards another :class:`PbrMaterial`, the parameters interpolate; towards
        any other material (an unlit one, say), it switches at ``alpha`` 1, as
        :meth:`Material.interpolate` does.
        """
        if not isinstance(end, PbrMaterial) or alpha >= 1:
            # Not super(): slotted dataclasses are new classes it does not find
            # before Python 3.14.
            return Material.interpolate(self, end, alpha)
        if alpha <= 0:
            return self
        return PbrMaterial(
            roughness=(1 - alpha) * self.roughness + alpha * end.roughness,
            metallic=(1 - alpha) * self.metallic + alpha * end.metallic,
            emissive=(
                (1 - alpha) * self.emissive[0] + alpha * end.emissive[0],
                (1 - alpha) * self.emissive[1] + alpha * end.emissive[1],
                (1 - alpha) * self.emissive[2] + alpha * end.emissive[2],
            ),
        )


def interpolate_material(start: Material, end: Material, alpha: float) -> Material:
    """The material at progress ``alpha`` from ``start`` to ``end``: ``end`` itself
    when they are equal or at ``alpha`` 1, ``start`` itself at ``alpha`` 0, else
    :meth:`Material.interpolate`.
    """
    if alpha >= 1 or start == end:
        return end
    if alpha <= 0:
        return start
    return start.interpolate(end, alpha)
