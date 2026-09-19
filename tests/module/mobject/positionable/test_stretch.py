"""Tests the `Positionable.stretch` method."""

import numpy as np
import pytest

from manim.mobject.abstract.positionable import Positionable
from manim.typing import Vector3DLike
from tests.module.mobject.positionable.utils import ANCHOR_POINTS, CUBE_VERTICES


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_no_points(dim: int) -> None:
    """Tests whether `stretch` works correctly for an object with no points."""
    p = Positionable()
    p.stretch(2, dim=dim, about_point=None, about_edge=None)
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


@pytest.mark.parametrize(
    ("dim", "expected"),
    [
        (0, [(-6, 1, 2), (3, 4, 5), (12, 7, 8)]),
        (1, [(0, -5, 2), (3, 4, 5), (6, 13, 8)]),
        (2, [(0, 1, -4), (3, 4, 5), (6, 7, 14)]),
    ],
)
def test_defaults(dim: int, expected: list[tuple[float, float, float]]) -> None:
    """Tests whether `stretch` works correctly for a single object when using the default parameters."""
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.stretch(3, dim=dim)
    np.testing.assert_allclose(p.points, expected)


@pytest.mark.parametrize(
    ("dim", "expected"),
    [
        (0, [(-2, 1, 2), (7, 4, 5), (16, 7, 8)]),
        (1, [(0, -1, 2), (3, 8, 5), (6, 17, 8)]),
        (2, [(0, 1, 0), (3, 4, 9), (6, 7, 18)]),
    ],
)
def test_about_point(dim: int, expected: list[tuple[float, float, float]]) -> None:
    """Tests whether `stretch` works correctly for a single object when passing the `about_point` parameter."""
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.stretch(3, dim=dim, about_point=(1, 2, 3))
    np.testing.assert_allclose(p.points, expected)


@pytest.mark.parametrize("factor", [-1.0, 0.0, 1.0, 2.0])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("about_edge", ANCHOR_POINTS)
def test_about_edge(factor: float, dim: int, about_edge: Vector3DLike) -> None:
    """Tests whether `stretch` works correctly for a cube when passing the `about_edge` parameter."""
    about_point = about_edge
    expected_points = CUBE_VERTICES.copy()
    expected_points -= about_point
    expected_points[:, dim] *= factor
    expected_points += about_point

    p = Positionable().set_points(CUBE_VERTICES)
    p.stretch(factor, dim=dim, about_edge=about_edge)
    np.testing.assert_allclose(p.points, expected_points)


def test_about_point_and_edge() -> None:
    """Tests whether the `stretch` method favors the `about_point` over the `about_edge` parameter."""
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.stretch(3, dim=0, about_point=(1, 2, 3), about_edge=(-1, 0, 1))
    np.testing.assert_allclose(p.points, [(-2, 1, 2), (7, 4, 5), (16, 7, 8)])

    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.stretch(3, dim=1, about_point=(1, 2, 3), about_edge=(-1, 0, 1))
    np.testing.assert_allclose(p.points, [(0, -1, 2), (3, 8, 5), (6, 17, 8)])

    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.stretch(3, dim=2, about_point=(1, 2, 3), about_edge=(-1, 0, 1))
    np.testing.assert_allclose(p.points, [(0, 1, 0), (3, 4, 9), (6, 7, 18)])
