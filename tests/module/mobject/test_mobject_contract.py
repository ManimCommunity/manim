"""State-level behavior of the common mobject classes.

These tests pin what the renderers draw from (points, colors, widths, draw order) and how
the common classes treat it, independent of any renderer.
"""

from __future__ import annotations

import numpy as np

from manim import (
    BLUE,
    GREEN,
    PI,
    RED,
    RIGHT,
    UP,
    UR,
    Arc,
    Circle,
    Group,
    ImageMobject,
    PMobject,
    Scene,
    Square,
    VGroup,
    VMobject,
)
from manim.utils.bezier import bezier
from manim.utils.color import color_to_rgba

# Geometry


def test_critical_points_use_anchors_and_sizes_use_all_control_points():
    # A half circle drawn as one cubic: anchors at y = 0, handles at y = 4/3.
    arc = Arc(angle=PI, num_components=2)
    assert len(arc.points) == 4
    np.testing.assert_allclose(arc.get_critical_point(UR), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(arc.height, 4 / 3)


def test_stroke_width_does_not_change_bounds():
    thick, thin = Circle(stroke_width=50), Circle(stroke_width=0)
    assert thick.width == thin.width == 2
    np.testing.assert_array_equal(thick.get_corner(UR), thin.get_corner(UR))


def test_bounds_follow_point_changes_in_children():
    child = Square(side_length=2)
    parent = VGroup(child)
    assert parent.width == 2
    child.points = child.points * 2
    assert parent.width == 4
    child.shift(RIGHT)
    np.testing.assert_allclose(parent.get_center(), RIGHT)


# Alignment (Transform)


def test_align_points_equalizes_counts_and_keeps_both_shapes():
    square, circle = Square(), Circle()
    square_corners = square.get_vertices().copy()
    circle_points = circle.points.copy()

    square.align_points(circle)

    assert len(square.points) == len(circle.points) == 32
    for corner in square_corners:
        assert np.isclose(square.points, corner).all(axis=1).any()
    # The square's curves are subdivided, not padded with zero-length curves.
    curve_lengths = np.linalg.norm(square.points[3::4] - square.points[::4], axis=1)
    np.testing.assert_allclose(curve_lengths, 1)
    np.testing.assert_array_equal(circle.points, circle_points)


def test_align_points_keeps_the_subpaths_of_the_richer_shape():
    two_parts = VMobject()
    two_parts.set_points_as_corners([[-1, 0, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0]])
    two_parts.add_subpath(Square(side_length=0.2).points)
    circle = Circle()

    two_parts.align_points(circle)

    assert len(two_parts.points) == len(circle.points)
    assert len(two_parts.get_subpaths()) == 2
    np.testing.assert_allclose([circle.width, circle.height], [2, 2])


def test_align_submobjects_keeps_originals_and_pads_with_copies():
    first, second = Square(), Square().shift(RIGHT)
    smaller = VGroup(first, second)
    larger = VGroup(*(Circle() for _ in range(3)))

    smaller.align_submobjects(larger)

    assert len(smaller.submobjects) == 3
    assert smaller.submobjects[0] is first
    assert smaller.submobjects[2] is second
    padding = smaller.submobjects[1]
    assert padding not in (first, second)
    np.testing.assert_allclose(padding.points, first.points)


# Partial curves (Create)


def test_partial_curve_at_zero_full_and_half_length():
    circle = Circle()  # 8 cubic curves
    partial = VMobject()

    partial.pointwise_become_partial(circle, 0, 0)
    np.testing.assert_allclose(partial.points, [circle.points[0]] * 4)

    partial.pointwise_become_partial(circle, 0, 1)
    np.testing.assert_allclose(partial.points, circle.points)

    partial.pointwise_become_partial(circle, 0, 0.5)
    np.testing.assert_allclose(partial.points[:16], circle.points[:16])
    np.testing.assert_allclose(partial.points[-1], circle.points[15])


def test_partial_curve_inside_one_curve_is_a_piece_of_that_curve():
    circle = Circle()
    partial = VMobject()

    partial.pointwise_become_partial(circle, 0.02, 0.1)

    first_curve = bezier(circle.points[:4])
    assert len(partial.points) == 4
    np.testing.assert_allclose(partial.points[0], first_curve(0.16))
    np.testing.assert_allclose(partial.points[-1], first_curve(0.8))


# Interpolation


def test_interpolate_is_linear_in_points_colors_and_width():
    start = Square(fill_opacity=1).set_color(RED).set_stroke(width=2)
    end = Square(fill_opacity=0).set_color([BLUE, GREEN]).set_stroke(width=6)
    end.shift(2 * RIGHT)
    start.align_data(end)
    middle = start.copy()

    middle.interpolate(start, end, 0.5)

    np.testing.assert_allclose(middle.points, (start.points + end.points) / 2)
    # Alignment spreads the single color to the length of the gradient.
    np.testing.assert_allclose(start.fill_rgbas, [color_to_rgba(RED)] * 2)
    np.testing.assert_allclose(
        middle.fill_rgbas, (start.fill_rgbas + end.fill_rgbas) / 2
    )
    assert len(middle.stroke_rgbas) == 2
    assert middle.stroke_width == 4
    assert middle.get_fill_opacity() == 0.5


# Copies


def test_copy_has_independent_points_and_colors():
    original = Square().set_color(RED)
    points, rgbas = original.points.copy(), original.stroke_rgbas.copy()

    copy = original.copy()
    copy.shift(UP).set_color(BLUE)

    np.testing.assert_array_equal(original.points, points)
    np.testing.assert_array_equal(original.stroke_rgbas, rgbas)


def test_copy_keeps_references_within_the_copied_family():
    shared, other = Circle(), Square()
    parent = VGroup(VGroup(shared, other), VGroup(shared))
    parent.highlighted = other
    parent.labels = ["a", ["b"]]

    copy = parent.copy()

    assert copy[0][0] is copy[1][0]
    assert copy[0][0] is not shared
    assert copy.highlighted is copy[0][1]
    assert copy.labels == parent.labels
    assert copy.labels[1] is not parent.labels[1]


# Draw order


def test_draw_order_is_family_order_stably_sorted_by_z_index(dry_run):
    scene = Scene()
    a, b, c, d, foreground = (Square() for _ in range(5))
    b.set_z_index(1)
    d.set_z_index(-1)
    scene.add(VGroup(a, b))
    scene.add_foreground_mobject(foreground)
    # Adding a group that contains a takes a out of its first group.
    scene.add(d, Group(c, a))

    drawn = scene.renderer.camera.get_mobjects_to_display(scene.mobjects)

    # Groups without points are not drawn; foreground mobjects come last only among
    # mobjects with the same z_index.
    assert drawn == [d, c, a, foreground, b]


# Point clouds and images


def test_point_cloud_set_color_colors_every_point():
    cloud = PMobject()
    cloud.add_points([[0, 0, 0], [1, 0, 0], [2, 0, 0]])

    cloud.set_color(RED)

    np.testing.assert_allclose(cloud.rgbas, [color_to_rgba(RED)] * 3)


def test_image_opacity_is_relative_to_the_original_alpha():
    image = ImageMobject(np.full((2, 2, 4), 200, dtype=np.uint8))

    image.set_opacity(0.5)
    image.set_opacity(0.5)
    assert (image.pixel_array[:, :, 3] == 100).all()

    image.set_color(BLUE)
    assert (image.pixel_array[:, :, 3] == 100).all()
    assert (image.pixel_array[:, :, :3] != 200).all()
