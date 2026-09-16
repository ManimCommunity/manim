import numpy as np
import pytest

from manim.mobject.abstract.positionable import Positionable
from manim.typing import Point3DLike, Vector3DLike


def test_no_points() -> None:
    p = Positionable()
    p.scale(2, about_point=None, about_edge=None)
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


def test_defaults() -> None:
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
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.scale(3, about_point=about_point)
    np.testing.assert_allclose(p.points, expected)


@pytest.mark.parametrize(
    argnames=("about_edge", "expected"),
    argvalues=[
        ((-1, -1, -1), [(-1, -1, -1), (5, 5, 5)]),
        ((-1, -1, +0), [(-1, -1, -3), (5, 5, 3)]),
        ((-1, -1, +1), [(-1, -1, -5), (5, 5, 1)]),
        ((-1, +0, -1), [(-1, -3, -1), (5, 3, 5)]),
        ((-1, +0, +0), [(-1, -3, -3), (5, 3, 3)]),
        ((-1, +0, +1), [(-1, -3, -5), (5, 3, 1)]),
        ((-1, +1, -1), [(-1, -5, -1), (5, 1, 5)]),
        ((-1, +1, +0), [(-1, -5, -3), (5, 1, 3)]),
        ((-1, +1, +1), [(-1, -5, -5), (5, 1, 1)]),
        ((+0, -1, -1), [(-3, -1, -1), (3, 5, 5)]),
        ((+0, -1, +0), [(-3, -1, -3), (3, 5, 3)]),
        ((+0, -1, +1), [(-3, -1, -5), (3, 5, 1)]),
        ((+0, +0, -1), [(-3, -3, -1), (3, 3, 5)]),
        ((+0, +0, +0), [(-3, -3, -3), (3, 3, 3)]),
        ((+0, +0, +1), [(-3, -3, -5), (3, 3, 1)]),
        ((+0, +1, -1), [(-3, -5, -1), (3, 1, 5)]),
        ((+0, +1, +0), [(-3, -5, -3), (3, 1, 3)]),
        ((+0, +1, +1), [(-3, -5, -5), (3, 1, 1)]),
        ((+1, -1, -1), [(-5, -1, -1), (1, 5, 5)]),
        ((+1, -1, +0), [(-5, -1, -3), (1, 5, 3)]),
        ((+1, -1, +1), [(-5, -1, -5), (1, 5, 1)]),
        ((+1, +0, -1), [(-5, -3, -1), (1, 3, 5)]),
        ((+1, +0, +0), [(-5, -3, -3), (1, 3, 3)]),
        ((+1, +0, +1), [(-5, -3, -5), (1, 3, 1)]),
        ((+1, +1, -1), [(-5, -5, -1), (1, 1, 5)]),
        ((+1, +1, +0), [(-5, -5, -3), (1, 1, 3)]),
        ((+1, +1, +1), [(-5, -5, -5), (1, 1, 1)]),
    ],
)
def test_about_edge(
    about_edge: Vector3DLike,
    expected: list[tuple[float, float, float]],
) -> None:
    p = Positionable().set_points([(-1, -1, -1), (1, 1, 1)])
    p.scale(3, about_edge=about_edge)
    np.testing.assert_allclose(p.points, expected)


def test_about_point_and_edge() -> None:
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.scale(3, about_point=(1, 2, 3), about_edge=(-1, 0, 1))
    np.testing.assert_allclose(p.points, [(-2, -1, 0), (7, 8, 9), (16, 17, 18)])
