"""Tests the `Positionable.get_anchor` method."""

import numpy as np
import pytest

from manim.mobject.abstract.positionable import Positionable
from manim.typing import Point3D, Vector3DLike
from tests.module.mobject.positionable.utils import (
    ANCHOR_POINTS,
    CUBE_VERTICES,
    POSITIONS,
)


def test_no_points_default() -> None:
    """Tests whether `get_anchor` with default parameters returns the origin point for an object with no points."""
    p = Positionable()
    np.testing.assert_allclose(p.get_anchor(), (0, 0, 0))


@pytest.mark.parametrize("anchor", ANCHOR_POINTS)
def test_no_points_anchor(anchor: Vector3DLike) -> None:
    """Tests whether `get_anchor` returns the origin point for any anchor for an object with no points."""
    p = Positionable()
    np.testing.assert_allclose(p.get_anchor(direction=anchor), (0, 0, 0))


def test_default() -> None:
    """Tests whether `get_anchor` returns the center anchor when no parameters are passed."""
    p = Positionable().set_points(CUBE_VERTICES)
    np.testing.assert_allclose(p.get_anchor(), (0, 0, 0))

    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    np.testing.assert_allclose(p.get_anchor(), (3, 4, 5))


@pytest.mark.parametrize("anchor", ANCHOR_POINTS)
def test_anchor_cube(anchor: Vector3DLike) -> None:
    """Tests whether `get_anchor` returns the correct position when passing the anchor parameter."""
    p = Positionable().set_points(CUBE_VERTICES)
    np.testing.assert_allclose(p.get_anchor(anchor), anchor)


@pytest.mark.parametrize("anchor", ANCHOR_POINTS)
@pytest.mark.parametrize("offset", POSITIONS)
def test_anchor_cube_with_offset(anchor: Vector3DLike, offset: Point3D) -> None:
    """Tests whether `get_anchor` returns the correct position when passing the anchor parameter."""
    p = Positionable().set_points(CUBE_VERTICES + offset)
    np.testing.assert_allclose(p.get_anchor(anchor), anchor + offset)
