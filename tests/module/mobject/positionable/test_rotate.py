import numpy as np
import pytest

from manim.constants import DEGREES
from manim.mobject.abstract.positionable import Positionable
from manim.typing import Point3DLike, Vector3DLike

ATOL = 1e-9


def test_no_points() -> None:
    p = Positionable()
    p.rotate(90 * DEGREES)
    np.testing.assert_allclose(
        p.points,
        np.zeros((0, 3)),
    )


def test_defaults() -> None:
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.rotate(90 * DEGREES)
    np.testing.assert_allclose(
        p.points,
        [(6, 1, 2), (3, 4, 5), (0, 7, 8)],
        atol=ATOL,
    )


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        ((0, 0, 0), [(0, 1, 2), (3, 4, 5), (6, 7, 8)]),
        ((1, 0, 0), [(0, 7, 2), (3, 4, 5), (6, 1, 8)]),
        ((0, 1, 0), [(0, 1, 8), (3, 4, 5), (6, 7, 2)]),
        ((0, 0, 1), [(6, 1, 2), (3, 4, 5), (0, 7, 8)]),
    ],
)
def test_axis(axis: Vector3DLike, expected: list[tuple[float, float, float]]) -> None:
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.rotate(90 * DEGREES, axis=axis)
    np.testing.assert_allclose(p.points, expected, atol=ATOL)


@pytest.mark.parametrize(
    ("about_point", "expected"),
    [
        ((0, 0, 0), [(-1, 0, 2), (-4, 3, 5), (-7, 6, 8)]),
        ((1, 2, 3), [(2, 1, 2), (-1, 4, 5), (-4, 7, 8)]),
    ],
)
def test_about_point(
    about_point: Point3DLike,
    expected: list[tuple[float, float, float]],
) -> None:
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.rotate(90 * DEGREES, about_point=about_point)
    np.testing.assert_allclose(p.points, expected, atol=ATOL)
