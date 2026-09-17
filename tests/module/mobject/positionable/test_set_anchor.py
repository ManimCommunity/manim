import numpy as np
import pytest

from manim.mobject.abstract.positionable import Positionable
from manim.typing import Point3D, Vector3DLike
from tests.module.mobject.positionable.utils import (
    ANCHOR_POINTS,
    CUBE_VERTICES,
    POSITIONS,
)


@pytest.mark.parametrize("position", POSITIONS)
def test_no_points_defaults(position: Point3D) -> None:
    p = Positionable()
    p.set_anchor(position)
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("anchor", ANCHOR_POINTS)
def test_no_points_anchor(position: Point3D, anchor: Vector3DLike) -> None:
    p = Positionable()
    p.set_anchor(position, direction=anchor)
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


@pytest.mark.parametrize("position", POSITIONS)
def test_cube_defaults(position: Point3D) -> None:
    expected_points = CUBE_VERTICES.copy() + position

    p = Positionable().set_points(CUBE_VERTICES)
    p.set_anchor(position)
    np.testing.assert_allclose(p.points, expected_points)


@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("anchor", ANCHOR_POINTS)
def test_cube_anchor(position: Point3D, anchor: Vector3DLike) -> None:
    expected_points = CUBE_VERTICES.copy() - anchor + position

    p = Positionable().set_points(CUBE_VERTICES)
    p.set_anchor(position, direction=anchor)
    np.testing.assert_allclose(p.points, expected_points)
