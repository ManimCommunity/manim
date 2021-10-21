from colour import Color

from manim import *
from tests.helpers.path_utils import get_svg_resource


def test_set_fill_color(using_opengl_renderer):
    expected_color = "#FF862F"
    svg = SVGMobject(get_svg_resource("heart.svg"), fill_color=expected_color)
    assert svg.fill_color == Color(expected_color)


def test_set_stroke_color(using_opengl_renderer):
    expected_color = "#FFFDDD"
    svg = SVGMobject(get_svg_resource("heart.svg"), stroke_color=expected_color)
    assert svg.stroke_color == Color(expected_color)


def test_set_color_sets_fill_and_stroke(using_opengl_renderer):
    expected_color = "#EEE777"
    svg = SVGMobject(get_svg_resource("heart.svg"), color=expected_color)
    assert svg.color == Color(expected_color)
    assert svg.fill_color == Color(expected_color)
    assert svg.stroke_color == Color(expected_color)


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
    assert svg.fill_color == Color(expected_color)


def test_stroke_overrides_color(using_opengl_renderer):
    expected_color = "#767676"
    svg = SVGMobject(
        get_svg_resource("heart.svg"),
        color="#334433",
        stroke_color=expected_color,
    )
    assert svg.stroke_color == Color(expected_color)
