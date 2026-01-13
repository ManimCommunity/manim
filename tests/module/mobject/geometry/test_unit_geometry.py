from __future__ import annotations

import logging

import numpy as np

from manim import (
    DEGREES,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    BackgroundRectangle,
    Circle,
    Line,
    Polygram,
    Sector,
    Square,
    SurroundingRectangle,
    TangentialArc,
)

logger = logging.getLogger(__name__)


def test_get_arc_center():
    np.testing.assert_array_equal(
        Sector(arc_center=[1, 2, 0]).get_arc_center(), [1, 2, 0]
    )


def test_Polygram_get_vertex_groups():
    # Test that, once a Polygram polygram is created with some vertex groups,
    # polygram.get_vertex_groups() (usually) returns the same vertex groups.
    vertex_groups_arr = [
        # 2 vertex groups for polygram 1
        [
            # Group 1: Triangle
            np.array(
                [
                    [2, 1, 0],
                    [0, 2, 0],
                    [-2, 1, 0],
                ]
            ),
            # Group 2: Square
            np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, -1, 0],
                ]
            ),
        ],
        # 3 vertex groups for polygram 1
        [
            # Group 1: Quadrilateral
            np.array(
                [
                    [2, 0, 0],
                    [0, -1, 0],
                    [0, 0, -2],
                    [0, 1, 0],
                ]
            ),
            # Group 2: Triangle
            np.array(
                [
                    [3, 1, 0],
                    [0, 0, 2],
                    [2, 0, 0],
                ]
            ),
            # Group 3: Pentagon
            np.array(
                [
                    [1, -1, 0],
                    [1, 1, 0],
                    [0, 2, 0],
                    [-1, 1, 0],
                    [-1, -1, 0],
                ]
            ),
        ],
    ]

    for vertex_groups in vertex_groups_arr:
        polygram = Polygram(*vertex_groups)
        poly_vertex_groups = polygram.get_vertex_groups()
        for poly_group, group in zip(poly_vertex_groups, vertex_groups, strict=False):
            np.testing.assert_array_equal(poly_group, group)

    # If polygram is a Polygram of a vertex group containing the start vertex N times,
    # then polygram.get_vertex_groups() splits it into N vertex groups.
    splittable_vertex_group = np.array(
        [
            [0, 1, 0],
            [1, -2, 0],
            [1, 2, 0],
            [0, 1, 0],  # same vertex as start
            [-1, 2, 0],
            [-1, -2, 0],
            [0, 1, 0],  # same vertex as start
            [0.5, 2, 0],
            [-0.5, 2, 0],
        ]
    )

    polygram = Polygram(splittable_vertex_group)
    assert len(polygram.get_vertex_groups()) == 3


def test_SurroundingRectangle():
    circle = Circle()
    square = Square()
    sr = SurroundingRectangle(circle, square)
    sr.set_style(fill_opacity=0.42)
    assert sr.get_fill_opacity() == 0.42


def test_TangentialArc():
    l1 = Line(start=LEFT, end=RIGHT)
    l2 = Line(start=DOWN, end=UP)
    l2.rotate(angle=45 * DEGREES, about_point=ORIGIN)
    arc = TangentialArc(l1, l2, radius=1.0)
    assert arc.radius == 1.0


def test_SurroundingRectangle_buff():
    sq = Square()
    rect1 = SurroundingRectangle(sq, buff=1)
    assert rect1.width == sq.width + 2
    assert rect1.height == sq.height + 2

    rect2 = SurroundingRectangle(sq, buff=(1, 2))
    assert rect2.width == sq.width + 2
    assert rect2.height == sq.height + 4


def test_BackgroundRectangle(manim_caplog):
    circle = Circle()
    square = Square()
    bg = BackgroundRectangle(circle, square)
    bg.set_style(fill_opacity=0.42)
    assert bg.get_fill_opacity() == 0.42
    bg.set_style(fill_opacity=1, hello="world")
    assert (
        "Argument {'hello': 'world'} is ignored in BackgroundRectangle.set_style."
        in manim_caplog.text
    )


def test_Square_side_length_reflets_correct_width_and_height():
    sq = Square(side_length=1).scale(3)
    assert sq.side_length == 3
    assert sq.height == 3
    assert sq.width == 3


def test_changing_Square_side_length_updates_the_square_appropriately():
    sq = Square(side_length=1)
    sq.side_length = 3
    assert sq.height == 3
    assert sq.width == 3


def test_Square_side_length_consistent_after_scale_and_rotation():
    sq = Square(side_length=1).scale(3).rotate(np.pi / 4)
    assert np.isclose(sq.side_length, 3)


def test_line_with_buff_and_path_arc():
    line = Line(LEFT, RIGHT, path_arc=60 * DEGREES, buff=0.3)
    expected_points = np.array(
        [
            [-0.7299265, -0.12999304, 0.0],
            [-0.6605293, -0.15719695, 0.0],
            [-0.58965623, -0.18050364, 0.0],
            [-0.51763809, -0.19980085, 0.0],
            [-0.51763809, -0.19980085, 0.0],
            [-0.43331506, -0.22239513, 0.0],
            [-0.34760317, -0.23944429, 0.0],
            [-0.26105238, -0.25083892, 0.0],
            [-0.26105238, -0.25083892, 0.0],
            [-0.1745016, -0.26223354, 0.0],
            [-0.08729763, -0.26794919, 0.0],
            [0.0, -0.26794919, 0.0],
            [0.0, -0.26794919, 0.0],
            [0.08729763, -0.26794919, 0.0],
            [0.1745016, -0.26223354, 0.0],
            [0.26105238, -0.25083892, 0.0],
            [0.26105238, -0.25083892, 0.0],
            [0.34760317, -0.23944429, 0.0],
            [0.43331506, -0.22239513, 0.0],
            [0.51763809, -0.19980085, 0.0],
            [0.51763809, -0.19980085, 0.0],
            [0.58965623, -0.18050364, 0.0],
            [0.6605293, -0.15719695, 0.0],
            [0.7299265, -0.12999304, 0.0],
        ]
    )
    np.testing.assert_allclose(line.points, expected_points)


def test_Circle_point_at_angle():
    from manim import TAU

    # Test basic angles
    circle = Circle(radius=1.0)
    start_point = circle.points[0].copy()

    # Angle 0 should return start point
    p0 = circle.point_at_angle(0)
    np.testing.assert_array_almost_equal(p0, start_point, decimal=5)

    # Angle π/2 should return point 90° along arc
    p90 = circle.point_at_angle(TAU / 4)
    # Verify it's approximately at top of circle
    assert p90[1] > 0.9  # Y coordinate close to 1

    # Angle π should return point opposite to start
    p180 = circle.point_at_angle(TAU / 2)
    # Verify it's opposite to start point
    np.testing.assert_array_almost_equal(p180[:2], -start_point[:2], decimal=5)

    # Angle beyond 2π should wrap around
    p720 = circle.point_at_angle(2 * TAU)
    np.testing.assert_array_almost_equal(p720[:2], start_point[:2], decimal=5)

    # Negative angles should work
    p_neg = circle.point_at_angle(-TAU / 4)
    # Should be same as 3/4 TAU
    p270 = circle.point_at_angle(3 * TAU / 4)
    np.testing.assert_array_almost_equal(p_neg, p270, decimal=5)

    # Test with rotated circle
    rotated_circle = Circle(radius=1.0).rotate(TAU / 8)
    rotated_start = rotated_circle.points[0].copy()

    # Angle 0 should still return start point after rotation
    p_rotated_0 = rotated_circle.point_at_angle(0)
    np.testing.assert_array_almost_equal(p_rotated_0, rotated_start, decimal=5)

    # Test with reflected circle
    reflected_circle = Circle(radius=1.0).flip()
    reflected_start = reflected_circle.points[0].copy()

    # Angle 0 should return start point even after reflection
    p_reflected_0 = reflected_circle.point_at_angle(0)
    np.testing.assert_array_almost_equal(p_reflected_0, reflected_start, decimal=5)
