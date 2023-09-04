from __future__ import annotations

from manim import *
from tests.helpers.path_utils import get_svg_resource


def test_set_fill_color():
    expected_color = "#FF862F"
    svg = SVGMobject(get_svg_resource("heart.svg"), fill_color=expected_color)
    assert svg.fill_color.to_hex() == expected_color


def test_set_stroke_color():
    expected_color = "#FFFDDD"
    svg = SVGMobject(get_svg_resource("heart.svg"), stroke_color=expected_color)
    assert svg.stroke_color.to_hex() == expected_color


def test_set_color_sets_fill_and_stroke():
    expected_color = "#EEE777"
    svg = SVGMobject(get_svg_resource("heart.svg"), color=expected_color)
    assert svg.color.to_hex() == expected_color
    assert svg.fill_color.to_hex() == expected_color
    assert svg.stroke_color.to_hex() == expected_color


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
    assert svg.fill_color.to_hex() == expected_color


def test_stroke_overrides_color():
    expected_color = "#767676"
    svg = SVGMobject(
        get_svg_resource("heart.svg"),
        color="#334433",
        stroke_color=expected_color,
    )
    assert svg.stroke_color.to_hex() == expected_color


def test_single_path_turns_into_sequence_of_points():
    svg = SVGMobject(
        get_svg_resource("cubic_and_lineto.svg"),
    )
    assert len(svg.points) == 0, svg.points
    assert len(svg.submobjects) == 1, svg.submobjects
    path = svg.submobjects[0]
    np.testing.assert_almost_equal(
        path.points,
        np.array(
            [
                [-0.166666666666666, 0.66666666666666, 0.0],
                [-0.166666666666666, 0.0, 0.0],
                [0.5, 0.66666666666666, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [-0.16666666666666666, 0.0, 0.0],
                [0.5, -0.6666666666666666, 0.0],
                [-0.166666666666666, -0.66666666666666, 0.0],
                [-0.166666666666666, -0.66666666666666, 0.0],
                [-0.27777777777777, -0.77777777777777, 0.0],
                [-0.38888888888888, -0.88888888888888, 0.0],
                [-0.5, -1.0, 0.0],
                [-0.5, -1.0, 0.0],
                [-0.5, -0.333333333333, 0.0],
                [-0.5, 0.3333333333333, 0.0],
                [-0.5, 1.0, 0.0],
                [-0.5, 1.0, 0.0],
                [-0.38888888888888, 0.8888888888888, 0.0],
                [-0.27777777777777, 0.7777777777777, 0.0],
                [-0.16666666666666, 0.6666666666666, 0.0],
            ]
        ),
        decimal=5,
    )


def test_closed_path_does_not_have_extra_point():
    # This dash.svg is the output of a "-" as generated from LaTex.
    # It ends back where it starts, so we shouldn't see a final line.
    svg = SVGMobject(
        get_svg_resource("dash.svg"),
    )
    assert len(svg.points) == 0, svg.points
    assert len(svg.submobjects) == 1, svg.submobjects
    dash = svg.submobjects[0]
    np.testing.assert_almost_equal(
        dash.points,
        np.array(
            [
                [13.524988331417841, -1.0, 0],
                [14.374988080480586, -1.0, 0],
                [15.274984567359079, -1.0, 0],
                [15.274984567359079, 0.0, 0.0],
                [15.274984567359079, 0.0, 0.0],
                [15.274984567359079, 1.0, 0.0],
                [14.374988080480586, 1.0, 0.0],
                [13.524988331417841, 1.0, 0.0],
                [13.524988331417841, 1.0, 0.0],
                [4.508331116720995, 1.0, 0],
                [-4.508326097975995, 1.0, 0.0],
                [-13.524983312672841, 1.0, 0.0],
                [-13.524983312672841, 1.0, 0.0],
                [-14.374983061735586, 1.0, 0.0],
                [-15.274984567359079, 1.0, 0.0],
                [-15.274984567359079, 0.0, 0.0],
                [-15.274984567359079, 0.0, 0.0],
                [-15.274984567359079, -1.0, 0],
                [-14.374983061735586, -1.0, 0],
                [-13.524983312672841, -1.0, 0],
                [-13.524983312672841, -1.0, 0],
                [-4.508326097975995, -1.0, 0],
                [4.508331116720995, -1.0, 0],
                [13.524988331417841, -1.0, 0],
            ]
        ),
        decimal=5,
    )
