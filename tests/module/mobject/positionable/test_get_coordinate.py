"""Tests the `Positionable.get_coordinate` method."""

import numpy as np
import pytest

from manim.mobject.abstract.positionable import Positionable
from manim.typing import Vector3DLike
from tests.module.mobject.positionable.utils import (
    ANCHOR_POINTS,
    CUBE_VERTICES,
    DIMENSIONS,
)


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_no_points_default(dim: int) -> None:
    """Tests whether `get_coordinate` returns 0 for an object with no points when the default parameters are used."""
    p = Positionable()
    np.testing.assert_allclose(p.get_coordinate(dim), 0)


@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("anchor", ANCHOR_POINTS)
def test_no_points_anchor(dim: int, anchor: Vector3DLike) -> None:
    """Tests whether `get_coordinate` returns 0 for an object with no points when an anchor is passed."""
    p = Positionable()
    np.testing.assert_allclose(p.get_coordinate(dim, direction=anchor), 0)


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_default(dim: int) -> None:
    """Tests whether `get_coordinate` returns the correct value at the object center when no anchor is passed."""
    p = Positionable().set_points(CUBE_VERTICES)
    np.testing.assert_allclose(p.get_coordinate(dim=dim), 0)

    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    np.testing.assert_allclose(p.get_coordinate(dim), (3, 4, 5)[dim])


@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("anchor", ANCHOR_POINTS)
def test_anchor(dim: int, anchor: Vector3DLike) -> None:
    """Tests whether `get_coordinate` returns the correst value when an anchor is passed."""
    p = Positionable().set_points(CUBE_VERTICES)
    np.testing.assert_allclose(p.get_coordinate(dim, anchor), anchor[dim])
