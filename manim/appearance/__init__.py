"""How mobjects look: their materials.

The names listed here can be imported from :mod:`manim.appearance` directly (and
from ``manim``).

Modules
=======

.. autosummary::
    :toctree: ../reference

    ~material
"""

from __future__ import annotations

from .material import Material, PbrMaterial, UnlitMaterial

__all__ = ["Material", "PbrMaterial", "UnlitMaterial"]
