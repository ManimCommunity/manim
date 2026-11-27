from __future__ import annotations

import math

import numpy as np
import pytest

from manim.appearance.material import PbrMaterial, UnlitMaterial
from manim.renderer._lighting import (
    AMBIENT,
    DEFAULT_LIGHT_INTENSITY,
    Light,
    linear_to_srgb,
    shade,
    srgb_to_linear,
)

LIGHT = Light(np.array([0.0, 0.0, 10.0]))
EYE = np.array([0.0, 0.0, 20.0])
UP_FACING = np.array([[0.0, 0.0, 1.0]])
ORIGIN = np.zeros((1, 3))
WHITE = np.ones((1, 3))


def _by_hand(base, n, point, eye, light, ambient, material):
    """The formula of plan 41, section 4.5, for one surface element."""
    base = srgb_to_linear(np.asarray(base))
    roughness = max(material.roughness, 0.03)
    s = 2 / roughness**4 - 2
    diffuse = base * (1 - material.metallic)
    f0 = 0.04 * (1 - material.metallic) + base * material.metallic
    out = np.asarray(material.emissive) + ambient * (diffuse + f0)
    n = n / np.linalg.norm(n)
    v = (eye - point) / np.linalg.norm(eye - point)
    if n @ v < 0:
        n = -n
    offset = light.position - point
    d2 = max(offset @ offset, 1e-4)
    to_light = offset / np.linalg.norm(offset)
    h = (to_light + v) / np.linalg.norm(to_light + v)
    radiance = np.asarray(light.rgb) * light.intensity / d2
    highlight = (n @ h) ** s if n @ h > 0 else 0
    out = out + radiance * max(n @ to_light, 0) * (
        diffuse / math.pi + f0 * (s + 2) / (2 * math.pi) * highlight
    )
    return linear_to_srgb(out)


@pytest.mark.parametrize(
    "material",
    [
        PbrMaterial(),
        PbrMaterial(roughness=0.2, metallic=0.7),
        PbrMaterial(roughness=1, emissive=(0.1, 0, 0)),
    ],
    ids=["default", "metal", "rough-emissive"],
)
def test_shade_follows_the_formula(material):
    base = np.array([[0.2, 0.5, 0.8]])
    normal = np.array([[0.3, -0.2, 1.0]])
    point = np.array([[1.0, 0.5, 0.0]])
    light = Light(np.array([4.0, 2.0, 6.0]), intensity=300)

    result = shade(base, normal, point, EYE, [light], AMBIENT, material)

    expected = _by_hand(base[0], normal[0], point[0], EYE, light, AMBIENT, material)
    np.testing.assert_allclose(result[0], expected, atol=1e-12)


def test_the_default_light_fully_lights_a_white_diffuse_surface():
    light = Light(np.array([0.0, 0.0, math.sqrt(230)]))
    lit = shade(WHITE, UP_FACING, ORIGIN, EYE, [light], 0, PbrMaterial(roughness=1))
    assert light.intensity == DEFAULT_LIGHT_INTENSITY
    # Lambert diffuse at full intensity, plus a small highlight that is clamped away.
    np.testing.assert_allclose(lit, [[1, 1, 1]])


def test_without_lights_or_material_nothing_is_shaded():
    base = np.array([[0.2, 0.5, 0.8]])
    np.testing.assert_array_equal(
        shade(base, UP_FACING, ORIGIN, EYE, [], AMBIENT, PbrMaterial()), base
    )
    np.testing.assert_array_equal(
        shade(base, UP_FACING, ORIGIN, EYE, [LIGHT], AMBIENT, UnlitMaterial()), base
    )


def test_surfaces_are_two_sided():
    base = np.array([[0.5, 0.5, 0.5]])
    front = shade(base, UP_FACING, ORIGIN, EYE, [LIGHT], AMBIENT, PbrMaterial())
    back = shade(base, -UP_FACING, ORIGIN, EYE, [LIGHT], AMBIENT, PbrMaterial())
    np.testing.assert_allclose(front, back)


@pytest.mark.parametrize(
    ("normal", "point", "eye", "light", "roughness"),
    [
        (UP_FACING, ORIGIN, EYE, LIGHT, 0.0),
        (UP_FACING, ORIGIN, EYE, Light(np.zeros(3)), 0.5),
        (np.zeros((1, 3)), ORIGIN, EYE, LIGHT, 0.5),
        (UP_FACING, ORIGIN, np.array([0.0, 0.0, -10.0]), LIGHT, 0.5),
    ],
    ids=["roughness 0", "light on the surface", "zero-area face", "zero half vector"],
)
def test_shade_is_finite_at_the_edges(normal, point, eye, light, roughness):
    result = shade(
        np.array([[0.4, 0.6, 0.8]]),
        normal,
        point,
        eye,
        [light],
        AMBIENT,
        PbrMaterial(roughness=roughness),
    )
    assert np.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()


def test_a_face_without_area_gets_ambient_and_emission_only():
    material = PbrMaterial(emissive=(0.05, 0.0, 0.0))
    base = np.array([[0.4, 0.6, 0.8]])
    result = shade(base, np.zeros((1, 3)), ORIGIN, EYE, [LIGHT], AMBIENT, material)
    linear = srgb_to_linear(base) * (1 - 0) * AMBIENT + AMBIENT * 0.04
    np.testing.assert_allclose(result, linear_to_srgb(linear + [0.05, 0, 0]))


def test_light_and_eye_opposite_give_no_highlight():
    # Light behind the surface from the eye's side: the half vector is zero.
    material = PbrMaterial(roughness=0.1)
    base = np.array([[0.4, 0.6, 0.8]])
    eye = np.array([0.0, 0.0, 10.0])
    light = Light(np.array([0.0, 0.0, -10.0]))
    result = shade(base, UP_FACING, ORIGIN, eye, [light], AMBIENT, material)
    ambient_only = linear_to_srgb(AMBIENT * (srgb_to_linear(base) + 0.04))
    np.testing.assert_allclose(result, ambient_only)


def test_a_point_light_interpolates_only_between_point_lights():
    from manim import PointLight, Square

    with pytest.raises(TypeError, match="two PointLights"):
        PointLight().interpolate_color(PointLight(), Square(), 0.5)


def test_interpolate_material_keeps_the_start_at_alpha_zero():
    from manim.appearance.material import interpolate_material

    start, end = PbrMaterial(roughness=0.1), PbrMaterial(roughness=0.9)
    assert interpolate_material(start, end, 0) is start
    calls = []

    class Recording(PbrMaterial):
        def interpolate(self, end, alpha):
            calls.append(alpha)
            return super().interpolate(end, alpha)

    recording = Recording(roughness=0.2)
    assert interpolate_material(recording, end, 0) is recording
    assert calls == []  # the start itself, without asking the material
    assert interpolate_material(start, end, 1) is end
    # Materials that cannot be blended switch at the end.
    assert interpolate_material(start, UnlitMaterial(), 0.5) is start
    assert interpolate_material(start, UnlitMaterial(), 1) == UnlitMaterial()
