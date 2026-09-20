"""Tests the `Positionable.set_dim_size` method."""

import numpy as np
import pytest

from manim.mobject.abstract.positionable import Positionable
from manim.typing import Point3D, Vector3D
from tests.module.mobject.positionable.utils import (
    ANCHOR_POINTS,
    CUBE_VERTICES,
    DIMENSIONS,
    POSITIONS,
    SIZES,
)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dim", DIMENSIONS)
def test_no_points_defaults(size: float, dim: int) -> None:
    """Tests whether `set_dim_size` works correctly for an object with no points when using the default parameters."""
    p = Positionable()
    p.set_dim_size(size, dim)
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("stretch", [False, True])
def test_no_points_stretch(size: float, dim: int, stretch: bool) -> None:
    """Tests whether `set_dim_size` works correctly for an object with no points when passing the `stretch` parameter."""
    p = Positionable()
    p.set_dim_size(size, dim, stretch=stretch)
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("about_point", POSITIONS)
def test_no_points_about_point(size: float, dim: int, about_point: Point3D) -> None:
    """Tests whether `set_dim_size` works correctly for an object with no points when passing the `about_point` parameter."""
    p = Positionable()
    p.set_dim_size(size, dim, about_point=about_point)
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("about_edge", ANCHOR_POINTS)
def test_no_points_about_edge(size: float, dim: int, about_edge: Vector3D) -> None:
    """Tests whether `set_dim_size` works correctly for an object with no points when passing the `about_edge` parameter."""
    p = Positionable()
    p.set_dim_size(size, dim, about_edge=about_edge)
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dim", DIMENSIONS)
def test_cube_defaults(size: float, dim: int) -> None:
    """Tests whether `set_dim_size` works correctly for a cube when using the default parameters."""
    expected_points = CUBE_VERTICES.copy()
    expected_points *= size / 2

    p = Positionable().set_points(CUBE_VERTICES)
    p.set_dim_size(size, dim)
    np.testing.assert_allclose(p.points, expected_points)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("stretch", [False, True])
def test_cube_stretch(size: float, dim: int, stretch: bool) -> None:
    """Tests whether `set_dim_size` works correctly for a cube when passing the `stretch` parameter."""
    expected_points = CUBE_VERTICES.copy()
    if not stretch:
        expected_points *= size / 2
    else:
        expected_points[:, dim] *= size / 2

    p = Positionable().set_points(CUBE_VERTICES)
    p.set_dim_size(size, dim, stretch=stretch)
    np.testing.assert_allclose(p.points, expected_points)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("about_point", POSITIONS)
def test_cube_about_point(size: float, dim: int, about_point: Point3D) -> None:
    """Tests whether `set_dim_size` works correctly for a cube when passing the `about_point` parameter."""
    expected_points = CUBE_VERTICES.copy()
    expected_points -= about_point
    expected_points *= size / 2
    expected_points += about_point

    p = Positionable().set_points(CUBE_VERTICES)
    p.set_dim_size(size, dim, about_point=about_point)
    np.testing.assert_allclose(p.points, expected_points)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("about_edge", ANCHOR_POINTS)
def test_cube_about_edge(size: float, dim: int, about_edge: Vector3D) -> None:
    """Tests whether `set_dim_size` works correctly for a cube when passing the `about_edge` parameter."""
    about_point = about_edge
    expected_points = CUBE_VERTICES.copy()
    expected_points -= about_point
    expected_points *= size / 2
    expected_points += about_point

    p = Positionable().set_points(CUBE_VERTICES)
    p.set_dim_size(size, dim, about_edge=about_edge)
    np.testing.assert_allclose(p.points, expected_points)
