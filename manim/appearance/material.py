"""How a surface responds to light.

A mobject's material says whether and how it is lit. Its color comes from the mobject
(its fill or stroke); the material adds how light falls on it. Materials are immutable
values: compare them with ``==``, and assign a new one to change a mobject's material
(:meth:`.VMobject.set_material`, :meth:`.MeshMobject.set_material`).

Meshes, surfaces and :class:`.ThreeDVMobject` are :class:`Pbr` by default; every other
mobject is :class:`Unlit`. Mobjects fixed in the frame are never lit, and without
lights nothing is shaded. See :doc:`/guides/colors_materials_lights`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Material", "PbrMaterial", "UnlitMaterial", "interpolate_material"]


@dataclass(frozen=True, slots=True)
class UnlitMaterial:
    """Not lit: drawn in its own color, whatever lights there are."""


@dataclass(frozen=True, slots=True)
class PbrMaterial:
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


Material = UnlitMaterial | PbrMaterial


def interpolate_material(start: Material, end: Material, alpha: float) -> Material:
    """The material at progress ``alpha`` from ``start`` to ``end``.

    Two :class:`.PbrMaterial` materials interpolate their parameters; at ``alpha`` 1 the
    result is ``end`` itself. Between other materials, the material switches to
    ``end`` when ``alpha`` reaches 1.
    """
    if alpha >= 1 or start == end:
        return end
    if isinstance(start, PbrMaterial) and isinstance(end, PbrMaterial):
        if alpha <= 0:
            return start
        return PbrMaterial(
            roughness=(1 - alpha) * start.roughness + alpha * end.roughness,
            metallic=(1 - alpha) * start.metallic + alpha * end.metallic,
            emissive=(
                (1 - alpha) * start.emissive[0] + alpha * end.emissive[0],
                (1 - alpha) * start.emissive[1] + alpha * end.emissive[1],
                (1 - alpha) * start.emissive[2] + alpha * end.emissive[2],
            ),
        )
    return start
