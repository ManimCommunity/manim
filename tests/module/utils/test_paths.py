from __future__ import annotations

import numpy as np
import pytest

from manim.constants import OUT, PI, TAU, UP
from manim.utils.bezier import interpolate
from manim.utils.paths import (
    STRAIGHT_PATH_THRESHOLD,
    clockwise_path,
    counterclockwise_path,
    path_along_arc,
    path_along_circles,
    spiral_path,
    straight_path,
)

START = np.array([[-1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, -1.0, 1.0]])
END = np.array([[1.0, 0.0, 0.0], [2.0, 2.0, 0.0], [-1.0, 4.0, 1.0]])

PATH_FUNCS = {
    "straight": straight_path(),
    "arc_positive": path_along_arc(PI / 3),
    "arc_negative": path_along_arc(-2 * PI / 3),
    "clockwise": clockwise_path(),
    "counterclockwise": counterclockwise_path(),
    "spiral_positive": spiral_path(TAU),
    "spiral_negative": spiral_path(-PI / 4),
    "circles": path_along_circles(PI / 2, np.array([0.0, -2.0, 0.0])),
}
parametrize_path_funcs = pytest.mark.parametrize(
    "path_func", PATH_FUNCS.values(), ids=PATH_FUNCS.keys()
)


@parametrize_path_funcs
def test_path_starts_at_start_points(path_func):
    np.testing.assert_allclose(path_func(START, END, 0), START, atol=1e-10)


@parametrize_path_funcs
def test_path_ends_at_end_points(path_func):
    np.testing.assert_allclose(path_func(START, END, 1), END, atol=1e-10)


@parametrize_path_funcs
def test_path_preserves_shape(path_func):
    assert path_func(START, END, 0.3).shape == START.shape


def test_straight_path_is_linear_interpolation():
    path = straight_path()
    assert path is interpolate
    np.testing.assert_allclose(path(START, END, 0.5), (START + END) / 2)


@pytest.mark.parametrize("make_path", [path_along_arc, spiral_path])
def test_tiny_angle_falls_back_to_straight_path(make_path):
    assert make_path(STRAIGHT_PATH_THRESHOLD / 2) is interpolate
    assert make_path(-STRAIGHT_PATH_THRESHOLD / 2) is interpolate
    assert make_path(STRAIGHT_PATH_THRESHOLD * 2) is not interpolate


def test_counterclockwise_path_goes_below_when_moving_right():
    start = np.array([[-1.0, 0.0, 0.0]])
    end = np.array([[1.0, 0.0, 0.0]])
    midpoint = counterclockwise_path()(start, end, 0.5)
    np.testing.assert_allclose(midpoint, [[0.0, -1.0, 0.0]], atol=1e-10)


def test_clockwise_path_goes_above_when_moving_right():
    start = np.array([[-1.0, 0.0, 0.0]])
    end = np.array([[1.0, 0.0, 0.0]])
    midpoint = clockwise_path()(start, end, 0.5)
    np.testing.assert_allclose(midpoint, [[0.0, 1.0, 0.0]], atol=1e-10)


@pytest.mark.parametrize("arc_angle", [PI / 3, PI / 2, PI, -3 * PI / 4])
def test_path_along_arc_stays_on_a_circle(arc_angle):
    # A chord of length 2 subtending an angle theta lies on a circle of
    # radius 1 / sin(theta / 2), centered on the chord's perpendicular bisector.
    start = np.array([[-1.0, 0.0, 0.0]])
    end = np.array([[1.0, 0.0, 0.0]])
    center = np.array([0.0, 1 / np.tan(arc_angle / 2), 0.0])
    radius = 1 / abs(np.sin(arc_angle / 2))
    path = path_along_arc(arc_angle)
    for alpha in np.linspace(0, 1, 11):
        point = path(start, end, alpha)
        assert np.linalg.norm(point - center) == pytest.approx(radius)


def test_path_along_arc_respects_axis():
    # Rotating about UP keeps every point's y-coordinate fixed.
    path = path_along_arc(PI / 2, axis=UP)
    start = np.array([[-1.0, 0.5, 0.0]])
    end = np.array([[1.0, 0.5, 0.0]])
    for alpha in np.linspace(0, 1, 5):
        assert path(start, end, alpha)[0, 1] == pytest.approx(0.5)
    np.testing.assert_allclose(path(start, end, 1), end, atol=1e-10)


def test_path_along_arc_zero_axis_falls_back_to_out():
    start = np.array([[-1.0, 0.0, 0.0]])
    end = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        path_along_arc(PI / 2, axis=np.zeros(3))(start, end, 0.5),
        path_along_arc(PI / 2, axis=OUT)(start, end, 0.5),
    )


def test_spiral_path_midpoint():
    # Halfway along, the point has covered half the displacement, rotated back
    # by half the spiral angle.
    start = np.array([[0.0, 0.0, 0.0]])
    end = np.array([[2.0, 0.0, 0.0]])
    midpoint = spiral_path(PI)(start, end, 0.5)
    np.testing.assert_allclose(midpoint, [[0.0, -1.0, 0.0]], atol=1e-10)


def test_path_along_circles_keeps_distance_to_center():
    # Start and end lie on the same circle around the center and are a quarter
    # turn apart, so the point should travel along that circle.
    center = np.array([1.0, 1.0, 0.0])
    start = np.array([[3.0, 1.0, 0.0]])
    end = np.array([[1.0, 3.0, 0.0]])
    path = path_along_circles(PI / 2, center)
    for alpha in np.linspace(0, 1, 9):
        point = path(start, end, alpha)
        assert np.linalg.norm(point - center) == pytest.approx(2.0)
