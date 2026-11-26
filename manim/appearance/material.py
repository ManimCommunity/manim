"""How a surface responds to light.

A mobject's material says whether and how it is lit. Its color comes from the mobject
(its fill or stroke); the material adds how light falls on it. Materials are immutable
values: compare them with ``==``, and assign a new one to change a mobject's material.

.. note::

    Mobjects cannot be given a material yet; renderers only use materials internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Material", "PbrMaterial", "UnlitMaterial"]


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
