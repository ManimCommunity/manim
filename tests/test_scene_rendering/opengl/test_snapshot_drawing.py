"""The OpenGL renderer draws the common mobject classes like the Cairo renderer."""

import numpy as np
import pytest

from manim import (
    BLUE,
    GREEN,
    LEFT,
    ORANGE,
    PI,
    PURE_BLUE,
    PURE_RED,
    RED,
    RIGHT,
    UP,
    Annulus,
    Arc,
    Circle,
    Line,
    MathTex,
    Scene,
    Square,
    Text,
    VMobject,
    tempconfig,
)


def _text():
    return [Text("Hello, snapshots", font_size=72)]


def _equation():
    return [MathTex(r"\int_0^1 x^2\,dx = \tfrac{1}{3}").scale(2)]


def _holes():
    return [Annulus(inner_radius=1, outer_radius=2, fill_opacity=0.8, color=BLUE)]


def _gradients():
    circle = Circle(2, fill_opacity=1, stroke_width=12)
    return [circle.set_fill([PURE_RED, PURE_BLUE]).set_stroke([GREEN, ORANGE])]


def _strokes():
    zigzag = VMobject(stroke_width=30).set_points_as_corners(
        [[-4, -1, 0], [-2, 1, 0], [0, -1, 0], [2, 1, 0], [4, -1, 0]]
    )
    return [zigzag]


def _open_subpaths():
    # Each open subpath is filled as if closed by a chord to its own start.
    path = VMobject(fill_opacity=1, stroke_width=0)
    path.points = np.concatenate(
        [
            Arc(1.5, angle=PI).shift(2.5 * LEFT).points,
            Arc(1.5, angle=PI).shift(2.5 * RIGHT + UP).points,
        ]
    )
    return [path]


def _thin_lines():
    return [
        Line([-6, y, 0], [6, y + 1, 0], stroke_width=1) for y in np.linspace(-3, 2, 6)
    ]


def _overlapping():
    # Drawn one after the other: the blue square must cover the red one.
    return [
        Square(3, fill_opacity=0.6, color=RED, stroke_width=6).shift(LEFT),
        Square(3, fill_opacity=0.6, color=BLUE, stroke_width=6).shift(RIGHT),
    ]


def _image(renderer, build):
    with tempconfig(
        {"renderer": renderer, "dry_run": True, "pixel_width": 384, "pixel_height": 216}
    ):
        scene = Scene()
        try:
            scene.add(*build())
            pixels = np.asarray(scene.get_image().convert("RGB"), dtype=float) / 255
        finally:
            scene.renderer.close()
    return pixels


@pytest.mark.parametrize(
    "build",
    [
        _text,
        _equation,
        _holes,
        _open_subpaths,
        _gradients,
        _strokes,
        _thin_lines,
        _overlapping,
    ],
    ids=lambda build: build.__name__.strip("_"),
)
def test_drawing_matches_cairo(opengl_common_mobjects, build):
    cairo = _image("cairo", build)
    opengl = _image("opengl", build)

    assert cairo.sum() > 0
    # Within 5%: sub-pixel lines come out about 4% lighter than Cairo's.
    assert opengl.sum() == pytest.approx(cairo.sum(), rel=0.05)
    assert np.abs(opengl - cairo).mean() < 0.01


@pytest.mark.parametrize("gap", [1.1, 2.2, 2.3, 3.0])
def test_fills_that_share_a_group_are_all_drawn(opengl_common_mobjects, gap):
    # Two squares that do not overlap are filled together, inside one rectangle of
    # pixels. Apple's software renderer (macOS without a GPU) lost one or both for
    # these distances when that rectangle was set with the scissor test.
    def build():
        return [
            Square(fill_opacity=1, color=RED, stroke_width=0).shift(gap * LEFT),
            Square(fill_opacity=1, color=BLUE, stroke_width=0).shift(gap * RIGHT),
        ]

    with tempconfig(
        {"renderer": "opengl", "dry_run": True, "pixel_width": 480, "pixel_height": 270}
    ):
        scene = Scene()
        try:
            scene.add(*build())
            pixels = np.asarray(scene.get_image().convert("RGB"))
        finally:
            scene.renderer.close()

    covered = pixels.max(axis=2) > 20
    # Each square is 2 of 8 units high: 67.5 pixels.
    assert covered[:, :240].sum() == pytest.approx(67.5**2, rel=0.03)
    assert covered[:, 240:].sum() == pytest.approx(67.5**2, rel=0.03)
