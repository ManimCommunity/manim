from __future__ import annotations

import numpy as np
import svgelements as se

from manim import *
from manim.mobject.svg.svg_mobject import VMobjectFromSVGPath
from tests.helpers.path_utils import get_svg_resource


def test_set_fill_color(using_opengl_renderer):
    expected_color = "#FF862F"
    svg = SVGMobject(get_svg_resource("heart.svg"), fill_color=expected_color)
    assert svg.fill_color.to_hex() == expected_color


def test_set_stroke_color(using_opengl_renderer):
    expected_color = "#FFFDDD"
    svg = SVGMobject(get_svg_resource("heart.svg"), stroke_color=expected_color)
    assert svg.stroke_color.to_hex() == expected_color


def test_set_color_sets_fill_and_stroke(using_opengl_renderer):
    expected_color = "#EEE777"
    svg = SVGMobject(get_svg_resource("heart.svg"), color=expected_color)
    assert svg.color.to_hex() == expected_color
    assert svg.fill_color.to_hex() == expected_color
    assert svg.stroke_color.to_hex() == expected_color


def test_set_fill_opacity(using_opengl_renderer):
    expected_opacity = 0.5
    svg = SVGMobject(get_svg_resource("heart.svg"), fill_opacity=expected_opacity)
    assert svg.fill_opacity == expected_opacity


def test_stroke_opacity(using_opengl_renderer):
    expected_opacity = 0.4
    svg = SVGMobject(get_svg_resource("heart.svg"), stroke_opacity=expected_opacity)
    assert svg.stroke_opacity == expected_opacity


def test_fill_overrides_color(using_opengl_renderer):
    expected_color = "#343434"
    svg = SVGMobject(
        get_svg_resource("heart.svg"),
        color="#123123",
        fill_color=expected_color,
    )
    assert svg.fill_color.to_hex() == expected_color


def test_stroke_overrides_color(using_opengl_renderer):
    expected_color = "#767676"
    svg = SVGMobject(
        get_svg_resource("heart.svg"),
        color="#334433",
        stroke_color=expected_color,
    )
    assert svg.stroke_color.to_hex() == expected_color


def test_subdivide_sharp_curves_under_opengl(using_opengl_renderer):
    # A fresh se.Path per construction: the constructor mutates path_obj in
    # place via approximate_arcs_with_quads.
    plain = VMobjectFromSVGPath(se.Path("M 0 0 Q 0 4 4 4 Z"))
    subdivided = VMobjectFromSVGPath(
        se.Path("M 0 0 Q 0 4 4 4 Z"),
        should_subdivide_sharp_curves=True,
    )
    assert len(subdivided.points) > len(plain.points)


def test_remove_null_curves_under_opengl(using_opengl_renderer):
    # The repeated "L 4 0" encodes a zero-length curve.
    plain = VMobjectFromSVGPath(se.Path("M 0 0 L 4 0 L 4 0 L 4 4 Z"))
    stripped = VMobjectFromSVGPath(
        se.Path("M 0 0 L 4 0 L 4 0 L 4 4 Z"),
        should_remove_null_curves=True,
    )
    assert len(stripped.points) < len(plain.points)


def test_long_lines_under_opengl(using_opengl_renderer):
    plain = VMobjectFromSVGPath(se.Path("M 0 0 L 4 0"))
    split = VMobjectFromSVGPath(
        se.Path("M 0 0 L 4 0"),
        long_lines=True,
    )
    assert len(split.points) == 2 * len(plain.points)


def test_path_flags_ignored_under_cairo(config):
    # The flags are OpenGL-only: under Cairo they must be accepted without
    # error and leave the generated points untouched.
    plain = VMobjectFromSVGPath(se.Path("M 0 0 Q 0 4 4 4 Z"))
    flagged = VMobjectFromSVGPath(
        se.Path("M 0 0 Q 0 4 4 4 Z"),
        long_lines=True,
        should_subdivide_sharp_curves=True,
        should_remove_null_curves=True,
    )
    assert np.array_equal(flagged.points, plain.points)
