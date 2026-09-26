"""Tests the `Positionable.rotate` method."""

import numpy as np
import pytest

from manim.constants import DEGREES
from manim.mobject.abstract.positionable import Positionable
from manim.typing import Point3D, Vector3D, Vector3DLike
from manim.utils.space_ops import rotation_matrix
from tests.module.mobject.positionable.utils import (
    ANCHOR_POINTS,
    ANGLES,
    CUBE_VERTICES,
    MAIN_AXES,
    POSITIONS,
)

ATOL = 1e-9
AXES = np.array([*MAIN_AXES, (0, 0, 0), (-0.4, 3, -100)])


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("axis", AXES)
def test_no_points(angle: float, axis: Vector3D) -> None:
    """Tests whether `rotate` works correctly for an object with no points."""
    p = Positionable()
    p.rotate(angle, axis)
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


def test_defaults() -> None:
    """Tests whether `rotate` works correctly for a simple object with default arguments."""
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.rotate(90 * DEGREES)
    np.testing.assert_allclose(p.points, [(6, 1, 2), (3, 4, 5), (0, 7, 8)], atol=ATOL)


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("axis", AXES)
def test_axis(angle: float, axis: Vector3DLike) -> None:
    """Tests whether the `axis` parameter of the `rotate` method works correctly."""
    expected_points = CUBE_VERTICES.copy()
    expected_points @= rotation_matrix(angle, axis).T

    p = Positionable().set_points(CUBE_VERTICES)
    p.rotate(angle, axis=axis)
    np.testing.assert_allclose(p.points, expected_points, atol=ATOL)


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("about_point", POSITIONS)
def test_about_point(angle: float, axis: Vector3D, about_point: Point3D) -> None:
    """Tests whether the `about_point` parameter of the `rotate` method works correctly."""
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
def test_about_edge(angle: float, axis: Vector3D, about_edge: Vector3D) -> None:
    """Tests whether the `about_edge` parameter of the `rotate` method works correctly."""
    about_point = about_edge
    expected_points = CUBE_VERTICES.copy()
    expected_points -= about_point
    expected_points @= rotation_matrix(angle, axis).T
    expected_points += about_point

    p = Positionable().set_points(CUBE_VERTICES)
    p.rotate(angle, axis, about_edge=about_edge)

    np.testing.assert_allclose(p.points, expected_points, atol=ATOL)


@pytest.mark.parametrize("angle", np.array([-5, -3, -1, 0, 1, 3, 5]) * 360 * DEGREES)
@pytest.mark.parametrize("axis", AXES)
def test_wraps_around(angle: int, axis: Vector3D) -> None:
    """Tests whether rotations of a multiple of 360 degrees do not affect the object."""
    p = Positionable().set_points(CUBE_VERTICES)
    p.rotate(angle, axis)
    np.testing.assert_allclose(p.points, CUBE_VERTICES, atol=ATOL)
