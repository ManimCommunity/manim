from __future__ import annotations

import numpy as np
import pytest

from manim import Axes, NumberPlane, ThreeDAxes

OBLIQUE_MATRIX = np.array([[1, 0.5, 0.2], [0.1, 1, 0.3], [0.2, 0, 1]])


def _make_axes(axes_class, **kwargs):
    options = {
        "x_range": (-3, 3, 1),
        "y_range": (-2, 2, 1),
        "x_length": 9,
        "y_length": 3,
        "tips": False,
    }
    if axes_class is ThreeDAxes:
        options.update(z_range=(-2, 3, 1), z_length=5)
    options.update(kwargs)
    return axes_class(**options)


def test_point_to_coords_oblique_plane_issue_3122(dry_run):
    plane = NumberPlane()
    plane.apply_matrix([[1, 2 / np.sqrt(13)], [0, 3 / np.sqrt(13)]])
    point = plane.c2p(1, 1)

    np.testing.assert_allclose(point, [1 + 2 / np.sqrt(13), 3 / np.sqrt(13), 0])
    np.testing.assert_allclose(plane.p2c(point), [1, 1], atol=1e-8)


@pytest.mark.parametrize("axes_class", [Axes, NumberPlane, ThreeDAxes])
@pytest.mark.parametrize("oblique", [False, True])
def test_point_to_coords_transformed_axes(dry_run, axes_class, oblique):
    axes = _make_axes(axes_class)
    if oblique:
        axes.apply_matrix(OBLIQUE_MATRIX)
    axes.rotate(0.4, axis=(1, 1, 1)).scale(0.7).shift((0.5, -1, 0.4))
    coords = np.array([1.25, -0.5, 0.75] if axes_class is ThreeDAxes else [1.25, -0.5])
    samples = np.array([coords, np.zeros_like(coords), -coords, 2 * coords])
    points = np.array([axes.c2p(*sample) for sample in samples])
    original_points = points.copy()

    np.testing.assert_allclose(axes.p2c(points[0]), coords, atol=1e-8)
    np.testing.assert_allclose(axes.p2c(points), samples, atol=1e-8)
    np.testing.assert_array_equal(points, original_points)


@pytest.mark.parametrize(
    ("x_range", "y_range"),
    [
        ((2, 6, 1), (3, 7, 1)),
        ((2, 6, 1), (-7, -3, 1)),
        ((-6, -2, 1), (3, 7, 1)),
        ((-6, -2, 1), (-7, -3, 1)),
    ],
)
def test_point_to_coords_oblique_axes_with_nonzero_ranges(dry_run, x_range, y_range):
    axes = _make_axes(Axes, x_range=x_range, y_range=y_range)
    coords = [x_range[0] + 2.5, y_range[0] + 1.5]
    np.testing.assert_allclose(axes.p2c(axes.c2p(*coords)), coords, atol=1e-8)
    axes.apply_matrix(OBLIQUE_MATRIX).shift((-0.7, 0.9, 0.2))

    np.testing.assert_allclose(axes.p2c(axes.c2p(*coords)), coords, atol=1e-8)


@pytest.mark.parametrize("count", [0, 1, 3, 4])
def test_point_to_coords_oblique_batch_shapes(dry_run, count):
    plane = _make_axes(NumberPlane).apply_matrix(OBLIQUE_MATRIX)
    coords = np.array([[1, 2], [-0.5, 1.5], [0, 0], [2, -1]])[:count]
    points = np.array([plane.c2p(*sample) for sample in coords]).reshape(-1, 3)

    result = plane.p2c(points)

    assert result.shape == (count, 2)
    np.testing.assert_allclose(result, coords, atol=1e-8)


@pytest.mark.parametrize("oblique", [False, True])
def test_point_to_coords_ignores_perpendicular_displacement(dry_run, oblique):
    axes = _make_axes(Axes)
    if oblique:
        axes.apply_matrix(OBLIQUE_MATRIX)
    axes.rotate(0.5, axis=(1, 1, 1)).shift((1, -0.5, 0.25))
    normal = np.cross(
        axes.x_axis.get_end() - axes.x_axis.get_start(),
        axes.y_axis.get_end() - axes.y_axis.get_start(),
    )
    normal /= np.linalg.norm(normal)
    point = axes.c2p(1, 1) + 0.7 * normal

    np.testing.assert_allclose(axes.p2c(point), [1, 1], atol=1e-8)
