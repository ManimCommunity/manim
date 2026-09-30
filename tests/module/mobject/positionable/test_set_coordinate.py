"""Tests the `Positionable.set_coordinate` method."""

import numpy as np
import pytest

from manim.mobject.abstract.positionable import Positionable
from manim.typing import Point3D, Vector3DLike
from tests.module.mobject.positionable.utils import (
    ANCHOR_POINTS,
    CUBE_VERTICES,
    DIMENSIONS,
    POSITIONS,
)


@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("dim", DIMENSIONS)
def test_no_points_defaults(position: Point3D, dim: int) -> None:
    """Tests whether `set_coordinate` works correctly for an object with no points when using the default parameters."""
    p = Positionable()
    p.set_coordinate(position[dim], dim)
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("anchor", ANCHOR_POINTS)
def test_no_points_anchor(position: Point3D, dim: int, anchor: Vector3DLike) -> None:
    """Tests whether `set_coordinate` works correctly for an object with no points when passing an anchor."""
    p = Positionable()
    p.set_coordinate(position[dim], dim, direction=anchor)
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("dim", DIMENSIONS)
def test_cube_defaults(position: Point3D, dim: int) -> None:
    """Tests whether `set_coordinate` works correctly for a cube when using the default parameters."""
    expected_points = CUBE_VERTICES.copy()
    expected_points[:, dim] += position[dim]

    p = Positionable().set_points(CUBE_VERTICES)
    p.set_coordinate(position[dim], dim)
    np.testing.assert_allclose(p.points, expected_points)


@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("anchor", ANCHOR_POINTS)
def test_cube_anchor(position: Point3D, dim: int, anchor: Vector3DLike) -> None:
    """Tests whether `set_coordinate` works correctly for a cube when passing an anchor."""
    expected_points = CUBE_VERTICES.copy()
    expected_points[:, dim] += position[dim] - anchor[dim]

    p = Positionable().set_points(CUBE_VERTICES)
    p.set_coordinate(position[dim], dim, direction=anchor)
    np.testing.assert_allclose(p.points, expected_points)
