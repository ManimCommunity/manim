"""Tests the `Positionable.scale` method."""

import numpy as np
import pytest

from manim.mobject.abstract.positionable import Positionable
from manim.typing import Point3DLike, Vector3DLike
from tests.module.mobject.positionable.utils import ANCHOR_POINTS, CUBE_VERTICES


@pytest.mark.parametrize("factor", [-1, 0, 1, 2, 5])
def test_no_points(factor: float) -> None:
    """Tests whether `scale` works correctly for an object with no points."""
    p = Positionable()
    p.scale(factor)
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


def test_defaults() -> None:
    """Tests whether `scale` defaults to scaling about the objects center."""
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.scale(3)
    np.testing.assert_allclose(p.points, [(-6, -5, -4), (3, 4, 5), (12, 13, 14)])


@pytest.mark.parametrize(
    ("about_point", "expected"),
    [
        ((0, 0, 0), [(0, 3, 6), (9, 12, 15), (18, 21, 24)]),
        ((1, 2, 3), [(-2, -1, 0), (7, 8, 9), (16, 17, 18)]),
    ],
)
def test_about_point(
    about_point: Point3DLike,
    expected: list[tuple[float, float, float]],
) -> None:
    """Tests whether the `about_point` parameter of the `scale` method works correctly."""
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.scale(3, about_point=about_point)
    np.testing.assert_allclose(p.points, expected)


@pytest.mark.parametrize("factor", [-1.0, 0.0, 1.0, 2.0])
@pytest.mark.parametrize("about_edge", ANCHOR_POINTS)
def test_about_edge(factor: float, about_edge: Vector3DLike) -> None:
    """Tests whether the `about_edge` parameter of the `scale` method works correctly."""
    about_point = about_edge
    expected_points = CUBE_VERTICES.copy()
    expected_points -= about_point
    expected_points *= factor
    expected_points += about_point

    p = Positionable().set_points(CUBE_VERTICES)
    p.scale(factor, about_edge=about_edge)
    np.testing.assert_allclose(p.points, expected_points)


def test_about_point_and_edge() -> None:
    """Tests whether the `scale` method favors the `about_point` over the `about_edge` parameter."""
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.scale(3, about_point=(1, 2, 3), about_edge=(-1, 0, 1))
    np.testing.assert_allclose(p.points, [(-2, -1, 0), (7, 8, 9), (16, 17, 18)])
