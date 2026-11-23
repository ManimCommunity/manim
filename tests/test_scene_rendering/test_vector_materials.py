"""Materials and depth ordering of vector mobjects (``material``, ``sort_by_depth``)."""

from __future__ import annotations

import numpy as np
import pytest

from manim import (
    DEGREES,
    OUT,
    PI,
    PURE_BLUE,
    PURE_RED,
    RIGHT,
    UP,
    Circle,
    Scene,
    Square,
    ThreeDScene,
    ThreeDVMobject,
    Transform,
    VGroup,
    tempconfig,
)
from manim.material import Pbr, Unlit
from manim.renderer.protocol import CapabilityError

RENDERERS = pytest.mark.parametrize("renderer", ["cairo", "opengl"])


def _settings(renderer):
    return tempconfig(
        {"renderer": renderer, "dry_run": True, "pixel_width": 96, "pixel_height": 54}
    )


def _image(renderer, build, scene_class=ThreeDScene):
    with _settings(renderer):
        scene = scene_class()
        if isinstance(scene, ThreeDScene):
            scene.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        scene.add(*build())
        try:
            return np.asarray(scene.get_image()).astype(int)
        finally:
            scene.renderer.close()


def _lit_square(angle=0.0):
    square = Square(3, fill_color=PURE_BLUE, fill_opacity=1, stroke_width=0)
    return square.rotate(angle, RIGHT).set_material(Pbr())


def _colours(pixels):
    """The colours of the drawn (not black) pixels."""
    drawn = pixels[..., :3][pixels[..., :3].sum(axis=-1) > 0]
    return {tuple(rgb) for rgb in drawn}


def test_vector_mobjects_are_unlit_and_unsorted_unless_asked():
    square = Square()
    assert square.material == Unlit()
    assert square.sort_by_depth is False
    group = VGroup(Square(), Circle())
    group.set_material(Pbr(roughness=0.1)).set_sort_by_depth(True)
    for mob in group.get_family():
        assert mob.material == Pbr(roughness=0.1)
        assert mob.sort_by_depth
    group.set_material(Unlit(), family=False)
    assert group.material == Unlit()
    assert group[0].material == Pbr(roughness=0.1)


def test_three_d_vmobjects_are_lit_and_sorted():
    mob = ThreeDVMobject()
    assert isinstance(mob.material, Pbr)
    assert mob.sort_by_depth


def test_material_and_depth_ordering_switch_at_the_end_of_a_transform():
    start = Square(fill_opacity=1)
    end = Square(fill_opacity=1).set_material(Pbr(roughness=0.1))
    end.set_sort_by_depth(True)
    animation = Transform(start, end)
    animation.begin()
    animation.interpolate(0.5)
    assert start.material == Unlit()
    assert not start.sort_by_depth
    animation.interpolate(1)
    assert start.material == Pbr(roughness=0.1)
    assert start.sort_by_depth

    lit = Square().set_material(Pbr(roughness=0))
    shinier = Square().set_material(Pbr(roughness=1))
    lit.interpolate_color(lit, shinier, 0.5)
    assert lit.material.roughness == pytest.approx(0.5)


@RENDERERS
def test_a_lit_path_is_one_colour_that_follows_its_plane(renderer):
    flat = _image(renderer, lambda: [_lit_square()])
    tilted = _image(renderer, lambda: [_lit_square(PI / 3)])
    unlit = _image(renderer, lambda: [_lit_square().set_material(Unlit())])
    # One colour inside (edge pixels blend with the background).
    inside = flat[24:31, 42:54, :3].reshape(-1, 3)
    assert len({tuple(rgb) for rgb in inside}) == 1
    assert tuple(flat[27, 48, :3]) != tuple(tilted[27, 48, :3])
    assert tuple(flat[27, 48, :3]) != tuple(unlit[27, 48, :3])


def test_cairo_and_opengl_light_a_path_alike():
    cairo = _image("cairo", lambda: [_lit_square(0.4)])[27, 48, :3]
    opengl = _image("opengl", lambda: [_lit_square(0.4)])[27, 48, :3]
    assert np.abs(cairo - opengl).max() <= 2


@RENDERERS
def test_lit_paths_without_lights_or_fixed_in_frame_are_unlit(renderer):
    unlit = _image(renderer, lambda: [_lit_square().set_material(Unlit())])
    # A plain Scene has no light.
    no_light = _image(renderer, lambda: [_lit_square()], Scene)
    fixed = _image(renderer, lambda: [_lit_square().fix_in_frame()])
    flat_unlit = _image(renderer, lambda: [_lit_square().set_material(Unlit())], Scene)
    np.testing.assert_array_equal(no_light, flat_unlit)
    assert tuple(fixed[27, 48, :3]) == tuple(flat_unlit[27, 48, :3])
    assert tuple(unlit[27, 48, :3]) == (0, 0, 255)


@RENDERERS
def test_strokes_of_lit_paths_are_not_lit(renderer):
    def square():
        return (
            Square(3, fill_color=PURE_BLUE, fill_opacity=1)
            .set_stroke(PURE_RED, width=60)
            .rotate(0.5, RIGHT)
            .set_material(Pbr())
        )

    pixels = _image(renderer, lambda: [square()])
    # The fill is lit (not pure blue); the stroke keeps its colour.
    assert (255, 0, 0) in _colours(pixels)
    assert tuple(pixels[27, 48, :3]) != (0, 0, 255)


@RENDERERS
def test_lit_paths_that_cannot_be_one_colour_are_drawn_without_lights(renderer):
    gradient = Square(fill_opacity=1).set_fill([PURE_RED, PURE_BLUE])
    pixels = _image(renderer, lambda: [gradient.set_material(Pbr())], Scene)
    assert len(_colours(pixels)) > 10


@RENDERERS
@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: Square(fill_opacity=1).set_fill([PURE_RED, PURE_BLUE]),
            "solid colour",
        ),
        (
            lambda: Square(fill_opacity=1).apply_function(
                lambda point: point + (point[0] * point[1]) * OUT
            ),
            "flat",
        ),
    ],
    ids=["gradient", "not flat"],
)
def test_lit_paths_that_cannot_be_one_colour_raise(renderer, build, message):
    with pytest.raises(CapabilityError, match=message):
        _image(renderer, lambda: [build().set_material(Pbr())])


@RENDERERS
def test_depth_ordering_draws_the_nearer_path_last(renderer):
    def build(sort):
        # Added first, but nearer the camera.
        near = Square(2, fill_color=PURE_RED, fill_opacity=1).shift(2 * OUT)
        far = Square(2, fill_color=PURE_BLUE, fill_opacity=1)
        for square in (near, far):
            square.set_sort_by_depth(sort)
        return [near, far]

    def centre(sort):
        with _settings(renderer):
            scene = ThreeDScene()
            scene.set_camera_orientation(phi=0, theta=-90 * DEGREES)
            scene.add(*build(sort))
            scene.camera.frame.shift(0.01 * UP)
            try:
                return np.asarray(scene.get_image())[27, 48, :3].astype(int)
            finally:
                scene.renderer.close()

    assert centre(True)[0] > 200
    assert centre(False)[2] > 200
