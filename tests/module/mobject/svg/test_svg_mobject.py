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


def test_close_command_closes_last_move_not_the_starting_one():
    # This A.svg is the output of a Text("A") in some systems
    # It contains a path that moves from the outer boundary of the A
    # to the boundary of the inner triangle, and then closes the path
    # which should close the inner triangle and not the outer boundary.
    svg = SVGMobject(
        get_svg_resource("A.svg"),
    )
    assert len(svg.points) == 0, svg.points
    assert len(svg.submobjects) == 1, svg.submobjects
    capital_A = svg.submobjects[0]

    # The last point should not be the same as the first point
    assert not all(capital_A.points[0] == capital_A.points[-1])
    np.testing.assert_almost_equal(
        capital_A.points,
        np.array(
            [
                [-0.8380339075214888, -1.0, 1.2246467991473532e-16],
                [-0.6132152047642527, -0.3333333333333336, 4.082155997157847e-17],
                [-0.388396502007016, 0.3333333333333336, -4.082155997157847e-17],
                [-0.16357779924977994, 1.0, -1.2246467991473532e-16],
                [-0.16357779924977994, 1.0, -1.2246467991473532e-16],
                [-0.05425733591657368, 1.0, -1.2246467991473532e-16],
                [0.05506312741663405, 1.0, -1.2246467991473532e-16],
                [0.16438359074984032, 1.0, -1.2246467991473532e-16],
                [0.16438359074984032, 1.0, -1.2246467991473532e-16],
                [0.3889336963403905, 0.3333333333333336, -4.082155997157847e-17],
                [0.6134838019309422, -0.3333333333333336, 4.082155997157847e-17],
                [0.8380339075214923, -1.0, 1.2246467991473532e-16],
                [0.8380339075214923, -1.0, 1.2246467991473532e-16],
                [0.744560897060354, -1.0, 1.2246467991473532e-16],
                [0.6510878865992157, -1.0, 1.2246467991473532e-16],
                [0.5576148761380774, -1.0, 1.2246467991473532e-16],
                [0.5576148761380774, -1.0, 1.2246467991473532e-16],
                [0.49717968849274957, -0.8138597980824822, 9.966907966764229e-17],
                [0.4367445008474217, -0.6277195961649644, 7.687347942054928e-17],
                [0.3763093132020939, -0.4415793942474466, 5.407787917345625e-17],
                [0.3763093132020939, -0.4415793942474466, 5.407787917345625e-17],
                [0.12167600863867864, -0.4415793942474466, 5.407787917345625e-17],
                [-0.13295729592473662, -0.4415793942474466, 5.407787917345625e-17],
                [-0.38759060048815186, -0.4415793942474466, 5.407787917345625e-17],
                [-0.38759060048815186, -0.4415793942474466, 5.407787917345625e-17],
                [-0.4480257881334797, -0.6277195961649644, 7.687347942054928e-17],
                [-0.5084609757788076, -0.8138597980824822, 9.966907966764229e-17],
                [-0.5688961634241354, -1.0, 1.2246467991473532e-16],
                [-0.5688961634241354, -1.0, 1.2246467991473532e-16],
                [-0.6586087447899202, -1.0, 1.2246467991473532e-16],
                [-0.7483213261557048, -1.0, 1.2246467991473532e-16],
                [-0.8380339075214888, -1.0, 1.2246467991473532e-16],
                [0.3021757525699033, -0.21434317946653003, 2.6249468865275272e-17],
                [0.1993017037512583, 0.09991949373745423, -1.2236608817799732e-17],
                [0.09642765493261184, 0.4141821669414385, -5.072268650087473e-17],
                [-0.006446393886033166, 0.7284448401454228, -8.920876418394973e-17],
                [-0.006446393886033166, 0.7284448401454228, -8.920876418394973e-17],
                [-0.10905185929034443, 0.4141821669414385, -5.072268650087473e-17],
                [-0.2116573246946542, 0.09991949373745423, -1.2236608817799732e-17],
                [-0.31426279009896546, -0.21434317946653003, 2.6249468865275272e-17],
                [-0.31426279009896546, -0.21434317946653003, 2.6249468865275272e-17],
                [-0.10878327587600921, -0.21434317946653003, 2.6249468865275272e-17],
                [0.09669623834694704, -0.21434317946653003, 2.6249468865275272e-17],
                [0.3021757525699033, -0.21434317946653003, 2.6249468865275272e-17],
            ]
        ),
        decimal=5,
    )
