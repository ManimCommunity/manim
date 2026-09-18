"""Tests the `Positionable.set_points` method."""

import numpy as np

from manim.mobject.abstract.positionable import Positionable
from tests.module.mobject.positionable.utils import PositionableWithFamily


def test_no_points() -> None:
    """Tests whether `set_points` works correctly for an object with no points."""
    p = Positionable()
    p.set_points([])
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


def test_single() -> None:
    """Tests whether `set_points` works correctly for an object without family members."""
    p = Positionable()
    p.set_points([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    np.testing.assert_allclose(p.points, [(1, 2, 3), (4, 5, 6), (7, 8, 9)])


def test_family() -> None:
    """Tests whether `set_points` does not affect family members."""
    p = PositionableWithFamily([Positionable(), Positionable(), Positionable()])
    p.set_points([(10, 11, 12), (13, 14, 15), (16, 17, 18)])
    np.testing.assert_allclose(p.points, [(10, 11, 12), (13, 14, 15), (16, 17, 18)])
    np.testing.assert_allclose(p.submobjects[0].points, np.zeros((0, 3)))
    np.testing.assert_allclose(p.submobjects[1].points, np.zeros((0, 3)))
    np.testing.assert_allclose(p.submobjects[2].points, np.zeros((0, 3)))
