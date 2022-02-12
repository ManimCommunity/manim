from __future__ import annotations

from colour import Color

from manim import *
from manim.mobject.svg.svg_path import string_to_numbers
from tests.helpers.path_utils import get_svg_resource


def test_set_fill_color():
    expected_color = "#FF862F"
    svg = SVGMobject(get_svg_resource("heart.svg"), fill_color=expected_color)
    assert svg.fill_color == Color(expected_color)


def test_set_stroke_color():
    expected_color = "#FFFDDD"
    svg = SVGMobject(get_svg_resource("heart.svg"), stroke_color=expected_color)
    assert svg.stroke_color == Color(expected_color)


def test_set_color_sets_fill_and_stroke():
    expected_color = "#EEE777"
    svg = SVGMobject(get_svg_resource("heart.svg"), color=expected_color)
    assert svg.color == Color(expected_color)
    assert svg.fill_color == Color(expected_color)
    assert svg.stroke_color == Color(expected_color)


def test_set_fill_opacity():
    expected_opacity = 0.5
    svg = SVGMobject(get_svg_resource("heart.svg"), fill_opacity=expected_opacity)
    assert svg.fill_opacity == expected_opacity


def test_stroke_opacity():
    expected_opacity = 0.4
    svg = SVGMobject(get_svg_resource("heart.svg"), stroke_opacity=expected_opacity)
    assert svg.stroke_opacity == expected_opacity


def test_fill_overrides_color():
    expected_color = "#343434"
    svg = SVGMobject(
        get_svg_resource("heart.svg"),
        color="#123123",
        fill_color=expected_color,
    )
    assert svg.fill_color == Color(expected_color)


def test_stroke_overrides_color():
    expected_color = "#767676"
    svg = SVGMobject(
        get_svg_resource("heart.svg"),
        color="#334433",
        stroke_color=expected_color,
    )
    assert svg.stroke_color == Color(expected_color)


def test_string_to_numbers():
    cases = {
        "3, 14, 159": [3.0, 14.0, 159.0],
        "2.7 1828-1828": [2.7, 1828.0, -1828.0],
        "1-.618.033": [1.0, -0.618, 0.033],
        # this one is a real-world example!
        "1.5938,1.5938,0,0,1-.1559.6874": [
            1.5938,
            1.5938,
            0.0,
            0.0,
            1.0,
            -0.1559,
            0.6874,
        ],
    }
    for case, result in cases.items():
        assert string_to_numbers(case) == result
