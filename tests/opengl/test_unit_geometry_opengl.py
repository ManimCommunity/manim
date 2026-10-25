from __future__ import annotations

import numpy as np

from manim import Arc, Sector


def test_get_arc_center(using_opengl_renderer):
    np.testing.assert_array_equal(
        Sector(arc_center=[1, 2, 0]).get_arc_center(), [1, 2, 0]
    )


def test_Arc_points(using_opengl_renderer):
    arc = Arc(radius=2, start_angle=0.5, angle=1.5, arc_center=[1, 2, 0])
    # OpenGL stores quadratic curves: num_components curves, 3 points per curve
    assert len(arc.points) == 3 * arc.num_components
    np.testing.assert_allclose(
        arc.get_start(), [1 + 2 * np.cos(0.5), 2 + 2 * np.sin(0.5), 0]
    )
    np.testing.assert_allclose(
        arc.get_end(), [1 + 2 * np.cos(2.0), 2 + 2 * np.sin(2.0), 0]
    )


def test_Arc_points_match_previous_output(using_opengl_renderer):
    arc = Arc(radius=2, start_angle=0.5, angle=1.5, arc_center=[1, 2, 0])
    # The quadratic points that init_points() produced before the refactor
    expected = Arc._create_quadratic_bezier_points(
        angle=1.5, start_angle=0.5, n_components=9
    ) * 2 + np.array([1, 2, 0])
    np.testing.assert_allclose(arc.points, expected)


def test_Arc_generate_points_is_idempotent(using_opengl_renderer):
    arc = Arc(radius=2, start_angle=0.5, angle=1.5, arc_center=[1, 2, 0])
    expected = arc.points.copy()
    arc.generate_points()
    np.testing.assert_array_equal(arc.points, expected)
