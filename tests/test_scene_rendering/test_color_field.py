"""Colour fields are sampled where renderers draw, on Cairo and OpenGL alike."""

from __future__ import annotations

import numpy as np
import pytest

from manim import (
    DEGREES,
    DOWN,
    LEFT,
    PURE_GREEN,
    RIGHT,
    UP,
    Camera,
    ColorField,
    ImageMobjectFromCamera,
    Line,
    ManimColor,
    MultiCamera,
    Scene,
    Square,
    ThreeDScene,
    tempconfig,
)
from manim.renderer.protocol import CapabilityError

RENDERERS = pytest.mark.parametrize("renderer", ["cairo", "opengl"])
W, H = 320, 180


def _settings(renderer, width=W, height=H):
    return tempconfig(
        {
            "renderer": renderer,
            "dry_run": True,
            "pixel_width": width,
            "pixel_height": height,
        }
    )


def _image(scene_class, mobjects, setup=None):
    scene = scene_class()
    scene.camera.background_opacity = 0
    if setup is not None:
        setup(scene)
    scene.add(*mobjects)
    try:
        return np.asarray(scene.get_image()).astype(int), scene.camera
    finally:
        scene.renderer.close()


def _band_field():
    """Blue over x = -4..4, with a narrow red band around x = 0.55."""
    size = 128
    xs = -4 + (np.arange(size) + 0.5) * 8 / size
    grid = np.zeros((1, size, 4))
    grid[..., 2] = grid[..., 3] = 1
    grid[0, np.abs(xs - 0.55) < 0.17] = (1, 0, 0, 1)
    return ColorField(grid, [[-4, 0.5, 0], [4, 0.5, 0], [-4, -0.5, 0]])


def _at(pixels, camera, x, y=0.0, width=W, height=H):
    column = int((0.5 + (x - camera.frame_center[0]) / camera.frame_width) * width)
    row = int((0.5 - (y - camera.frame_center[1]) / camera.frame_height) * height)
    return pixels[row, column]


@RENDERERS
@pytest.mark.parametrize("case", ["full", "trimmed", "panned", "fill"])
def test_a_narrow_band_shows_where_it_is(renderer, case):
    # Sampling the field at the geometry's points (not where it is drawn) loses
    # the band, and trimming would change the colour of points that did not move.
    line = Line(3 * LEFT, 3 * RIGHT, stroke_width=28)
    mobject = {
        "full": line,
        "trimmed": line.copy().pointwise_become_partial(line, 0.575, 1),
        "panned": line,
        "fill": Square(4, fill_opacity=1, stroke_width=0),
    }[case]

    def pan(scene):
        if case == "panned":
            scene.camera.frame.shift(1.1 * RIGHT)

    with _settings(renderer):
        solid, _ = _image(Scene, [mobject.copy()], pan)
        painted = mobject.copy()
        (painted.set_fill if case == "fill" else painted.set_stroke)(_band_field())
        field, camera = _image(Scene, [painted], pan)

    np.testing.assert_array_equal(_at(field, camera, 0.55)[:3], [255, 0, 0])
    if case != "trimmed":
        np.testing.assert_array_equal(_at(field, camera, -0.5)[:3], [0, 0, 255])
    # Coverage and alpha are those of the same shape in one opaque colour.
    np.testing.assert_array_equal(field[..., 3], solid[..., 3])


@RENDERERS
def test_the_sampler_fixture(renderer):
    # Red, green / blue, transparent white over -2..2; probes at texel centres,
    # edge midpoints, the middle and outside, against squares of the colour the
    # reference sampler gives.
    grid = np.array([[[1, 0, 0, 1], [0, 1, 0, 1]], [[0, 0, 1, 1], [1, 1, 1, 0]]], float)
    field = ColorField(grid, [[-2, 2, 0], [2, 2, 0], [-2, -2, 0]])
    probes = [(-1, 1), (1, 1), (-1, -1), (1, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
    probes += [(-3.5, 1), (1, -3.5)]

    def squares(use_field):
        result = []
        for x, y in probes:
            square = Square(0.2, stroke_width=0).move_to([x, y, 0])
            if use_field:
                square.set_fill(field, opacity=1)
            else:
                rgba = field.sample([[x, y, 0]])[0]
                square.set_fill(ManimColor.from_rgb(rgba[:3]), opacity=float(rgba[3]))
            result.append(square)
        return result

    with _settings(renderer, 640, 360):
        sampled, camera = _image(Scene, squares(True))
        reference, _ = _image(Scene, squares(False))
    for x, y in probes:
        got = _at(sampled, camera, x, y, 640, 360)
        want = _at(reference, camera, x, y, 640, 360)
        # 8-bit grids, textures and blending.
        np.testing.assert_allclose(got, want, atol=3, err_msg=str((x, y)))


@RENDERERS
def test_several_fields_and_solids_in_one_frame(renderer):
    red = ColorField([[[1, 0, 0, 1]]], [[-7, 4, 0], [7, 4, 0], [-7, -4, 0]])
    blue = ColorField(
        [[[0, 0, 1, 1], [0, 1, 1, 1]]], [[-7, 4, 0], [7, 4, 0], [-7, -4, 0]]
    )
    lines = [
        Line(3 * LEFT, 3 * RIGHT, stroke_width=30).shift(y * UP).set_stroke(paint)
        for y, paint in [(1, red), (0, blue), (-1, PURE_GREEN), (-2, red)]
    ]
    with _settings(renderer):
        pixels, camera = _image(Scene, lines)
    np.testing.assert_array_equal(_at(pixels, camera, 0, 1)[:3], [255, 0, 0])
    np.testing.assert_allclose(
        _at(pixels, camera, -2, 0)[:3],
        np.array(blue.sample([[-2, 0, 0]])[0][:3]) * 255,
        atol=3,
    )
    np.testing.assert_array_equal(_at(pixels, camera, 0, -1)[:3], [0, 255, 0])
    np.testing.assert_array_equal(_at(pixels, camera, 0, -2)[:3], [255, 0, 0])


@RENDERERS
def test_camera_views_sample_the_field_where_they_look(renderer):
    blue = ColorField(
        [[[0, 0, 1, 1], [0, 1, 1, 1]]], [[-7, 4, 0], [7, 4, 0], [-7, -4, 0]]
    )
    inner = Camera()
    inner.frame.scale(0.5)
    view = ImageMobjectFromCamera(inner).set(height=3).to_corner(UP + RIGHT)
    line = Line(3 * LEFT, 3 * RIGHT, stroke_width=30).set_stroke(blue)

    def with_view(scene):
        scene._camera = MultiCamera([view])

    with _settings(renderer):
        pixels, camera = _image(Scene, [line, view], with_view)
    x0, y0 = view.get_center()[:2]
    shown = _at(pixels, camera, x0 - view.width / inner.frame_width, y0)
    expected = np.array(blue.sample([[-1, 0, 0]])[0][:3]) * 255
    np.testing.assert_allclose(shown[:3], expected, atol=4)


@RENDERERS
@pytest.mark.parametrize("kind", ["field opacity", "translucent grid"])
def test_translucent_fields_cover_as_a_solid_of_the_same_alpha(renderer, kind):
    corners = [[-7, 4, 0], [7, 4, 0], [-7, -4, 0]]
    field = (
        ColorField([[[1, 0, 0, 1]]], corners, opacity=0.4)
        if kind == "field opacity"
        else ColorField([[[1, 0, 0, 0.4]]], corners)
    )

    def shapes(paint, opacity):
        line = Line(3 * LEFT, 3 * RIGHT, stroke_width=30).set_stroke(
            paint, opacity=opacity
        )
        square = (
            Square(2, stroke_width=0).shift(2 * DOWN).set_fill(paint, opacity=opacity)
        )
        return [line, square]

    with _settings(renderer):
        painted, _ = _image(Scene, shapes(field, None))
        solid, _ = _image(Scene, shapes("#FF0000", 0.4))
    np.testing.assert_array_equal(painted, solid)


def _perspective(scene):
    scene.set_camera_orientation(phi=60 * DEGREES)


@RENDERERS
@pytest.mark.parametrize(
    ("case", "scene_class", "setup", "corners", "fixed"),
    [
        ("perspective camera", ThreeDScene, _perspective, None, False),
        ("fixed in frame", Scene, None, None, True),
        (
            "field in another plane",
            Scene,
            None,
            [[-4, 1, 0], [4, 1, 1], [-4, -1, 0]],
            False,
        ),
    ],
)
def test_unsupported_fields_raise(renderer, case, scene_class, setup, corners, fixed):
    field = ColorField(
        [[[1, 0, 0, 1]]], corners or [[-4, 1, 0], [4, 1, 0], [-4, -1, 0]]
    )
    line = Line(3 * LEFT, 3 * RIGHT, stroke_width=20).set_stroke(field)
    if fixed:
        line.fix_in_frame()
    with _settings(renderer, 64, 36), pytest.raises(CapabilityError):
        _image(scene_class, [line], setup)


@RENDERERS
def test_an_image_of_a_field_scene_stays_valid_after_another(renderer):
    with _settings(renderer, 64, 36):
        scene = Scene()
        line = Line(3 * LEFT, 3 * RIGHT, stroke_width=30).set_stroke(_band_field())
        scene.add(line)
        try:
            first = np.asarray(scene.get_image()).copy()
            kept = scene.get_image()
            line.shift(DOWN)
            scene.get_image()
        finally:
            scene.renderer.close()
    np.testing.assert_array_equal(np.asarray(kept), first)
