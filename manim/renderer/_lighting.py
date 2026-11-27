"""How lit surfaces are colored: one shading function for every renderer.

:func:`shade` colors surfaces with a :class:`.PbrMaterial` under point lights, in linear
RGB, with the parameters and light units of glTF 2.0 and its
``KHR_lights_punctual`` extension. It approximates the native renderer's shading with a
Lambert diffuse and a normalized Blinn-Phong highlight in place of its GGX term, and no
tone mapping. The OpenGL mesh shader computes the same in GLSL
(``shaders/include/lighting.glsl``); keep the two in step.

Without lights, nothing is shaded: surfaces show their base color.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from manim.appearance.material import PbrMaterial

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manim.renderer.snapshot import Material

__all__ = [
    "AMBIENT",
    "DEFAULT_LIGHT_INTENSITY",
    "Light",
    "linear_to_srgb",
    "shade",
    "srgb_to_linear",
]

#: Intensity of the default light (candela, as in ``KHR_lights_punctual``): it lights a
#: white diffuse surface at the origin, facing the light at the default light position
#: (distance sqrt(230)), fully: ``pi * distance**2``.
DEFAULT_LIGHT_INTENSITY = math.pi * 230.0
#: Intensity of the uniform white light that reaches every surface.
AMBIENT = 0.25
#: Smallest roughness; at 0 the highlight's exponent divides by zero.
MIN_ROUGHNESS = 0.03
#: Smallest squared distance to a light (as the native renderer).
MIN_DISTANCE_SQUARED = 1e-4


class Light(NamedTuple):
    """A point light."""

    position: np.ndarray
    #: Linear RGB.
    rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = DEFAULT_LIGHT_INTENSITY


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=float), 0, 1)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)


def shade(
    base_rgb: np.ndarray,
    normals: np.ndarray,
    points: np.ndarray,
    eye: np.ndarray,
    lights: Sequence[Light],
    ambient: float,
    material: Material,
) -> np.ndarray:
    """The colors of surface elements, lit.

    Parameters
    ----------
    base_rgb
        sRGB base colors, shape ``(n, 3)``.
    normals
        Normals, shape ``(n, 3)``; need not be unit length, and are flipped towards
        the eye (surfaces are two-sided). A zero normal (a face without area) gets
        the emission and ambient light only.
    points
        Positions, shape ``(n, 3)``.
    eye
        Where the surfaces are seen from.
    lights
        The point lights.
    ambient
        Intensity of the uniform white light.
    material
        Only :class:`.PbrMaterial` materials are lit; any other material (also the
        provisional :class:`.LegacyShaded` of snapshots) is left as it is.

    Returns
    -------
    sRGB colors, shape ``(n, 3)``, clamped to ``[0, 1]``.
    """
    base_rgb = np.asarray(base_rgb, dtype=float)
    if not isinstance(material, PbrMaterial) or not lights:
        return base_rgb
    base = srgb_to_linear(base_rgb)
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    metallic = material.metallic
    roughness = max(material.roughness, MIN_ROUGHNESS)
    alpha = roughness**2
    exponent = 2 / alpha**2 - 2
    diffuse = base * (1 - metallic)
    f0 = 0.04 * (1 - metallic) + base * metallic
    out = np.asarray(material.emissive, dtype=float) + ambient * (diffuse + f0)

    # A face without area has a zero normal, so no light reaches it.
    n = _unit(normals)
    to_eye = _unit(np.asarray(eye, dtype=float) - points)
    # Two-sided: the side facing the eye.
    n = np.where(np.sum(n * to_eye, axis=1, keepdims=True) < 0, -n, n)
    for light in lights:
        offset = np.asarray(light.position, dtype=float) - points
        distance_squared = np.maximum(
            np.sum(offset * offset, axis=1, keepdims=True), MIN_DISTANCE_SQUARED
        )
        to_light = _unit(offset)
        radiance = (
            np.asarray(light.rgb, dtype=float) * light.intensity / distance_squared
        )
        n_dot_l = np.maximum(np.sum(n * to_light, axis=1, keepdims=True), 0)
        half = to_light + to_eye
        half_length = np.linalg.norm(half, axis=1, keepdims=True)
        n_dot_h = np.sum(n * half, axis=1, keepdims=True) / np.where(
            half_length > 0, half_length, 1
        )
        # No highlight without a half vector (light and eye opposite), nor where the
        # half vector is not in front of the surface (as the GLSL, where pow(0, 0)
        # is undefined).
        highlight = np.where(
            (half_length > 0) & (n_dot_h > 0), np.maximum(n_dot_h, 0) ** exponent, 0
        )
        specular = f0 * (exponent + 2) / (2 * math.pi) * highlight
        lit = radiance * n_dot_l * (diffuse / math.pi + specular)
        out = out + lit
    return linear_to_srgb(out)


def _unit(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)
