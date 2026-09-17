import numpy as np
import pytest

from manim.constants import DEGREES
from manim.mobject.abstract.positionable import Positionable
from manim.typing import Point3DLike, Vector3DLike
from manim.utils.space_ops import rotation_matrix
from tests.module.mobject.positionable.utils import ANCHOR_POINTS, AXES, CUBE_VERTICES

ATOL = 1e-9
ABOUT_POINTS = [(-3, -2, 1), (0, 0, 0), (1, 2, 3)]
ANGLES = [-360, -90, -45, -33, 0, 33, 45, 90, 360]


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


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("axis", AXES)
def test_axis(angle: float, axis: Vector3DLike) -> None:
    expected_points = CUBE_VERTICES.copy()
    expected_points @= rotation_matrix(angle, axis).T

    p = Positionable().set_points(CUBE_VERTICES)
    p.rotate(angle, axis=axis)
    np.testing.assert_allclose(p.points, expected_points, atol=ATOL)


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("about_point", ABOUT_POINTS)
def test_about_point(
    angle: float,
    axis: Vector3DLike,
    about_point: Point3DLike,
) -> None:
    expected_points = CUBE_VERTICES.copy()
    expected_points -= about_point
    expected_points @= rotation_matrix(angle, axis).T
    expected_points += about_point

    p = Positionable().set_points(CUBE_VERTICES)
    p.rotate(angle, axis, about_point=about_point)
    np.testing.assert_allclose(p.points, expected_points, atol=ATOL)


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("about_edge", ANCHOR_POINTS)
def test_about_edge(
    angle: float,
    axis: Vector3DLike,
    about_edge: Vector3DLike,
) -> None:
    about_point = about_edge
    expected_points = CUBE_VERTICES.copy()
    expected_points -= about_point
    expected_points @= rotation_matrix(angle, axis).T
    expected_points += about_point

    p = Positionable().set_points(CUBE_VERTICES)
    p.rotate(angle, axis, about_edge=about_edge)

    np.testing.assert_allclose(p.points, expected_points, atol=ATOL)
