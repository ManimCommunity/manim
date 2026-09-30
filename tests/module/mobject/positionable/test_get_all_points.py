"""Tests the `Positionable.get_all_points` method."""

import numpy as np

from manim.mobject.abstract.positionable import Positionable
from tests.module.mobject.positionable.utils import PositionableWithFamily


def test_no_points() -> None:
    """Tests whether `get_all_points` returns an empty point array when no points are present."""
    p = Positionable()
    np.testing.assert_allclose(p.get_all_points(), np.zeros((0, 3)))


def test_single() -> None:
    """Tests whether `get_all_points` works correctly for a single object without children."""
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    np.testing.assert_allclose(p.get_all_points(), [(0, 1, 2), (3, 4, 5), (6, 7, 8)])


def test_family() -> None:
    """Tests whether `get_all_points` works corrects for a object with children."""
    p = PositionableWithFamily(
        [
            Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)]),
            Positionable().set_points([(9, 10, 11), (12, 13, 14), (15, 16, 17)]),
            Positionable().set_points([(18, 19, 20), (21, 22, 23), (24, 25, 26)]),
        ]
    ).set_points([(27, 28, 29), (30, 31, 32), (33, 34, 35)])
    np.testing.assert_allclose(
        p.get_all_points(),
        [
            (27, 28, 29),
            (30, 31, 32),
            (33, 34, 35),
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (9, 10, 11),
            (12, 13, 14),
            (15, 16, 17),
            (18, 19, 20),
            (21, 22, 23),
            (24, 25, 26),
        ],
    )
