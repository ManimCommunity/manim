"""Tests the `Positionable.get_num_points` method."""

from manim.mobject.abstract.positionable import Positionable
from tests.module.mobject.positionable.utils import PositionableWithFamily


def test_no_points() -> None:
    """Tests whether `get_num_points` returns 0 for an object with no points."""
    p = Positionable()
    assert p.get_num_points() == 0


def test_single() -> None:
    """Tests whether `get_num_points` returns the correct count for a single object with no children."""
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    assert p.get_num_points() == 3


def test_family() -> None:
    """Tests whether `get_num_points` does not include family members."""
    p = PositionableWithFamily([Positionable(), Positionable(), Positionable()])
    assert p.get_num_points() == 0

    p = PositionableWithFamily(
        [
            Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)]),
            Positionable().set_points([(9, 10, 11), (12, 13, 14), (15, 16, 17)]),
            Positionable().set_points([(18, 19, 20), (21, 22, 23), (24, 25, 26)]),
        ]
    ).set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    assert p.get_num_points() == 3

    p = PositionableWithFamily(
        [
            Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)]),
            Positionable().set_points([(9, 10, 11), (12, 13, 14), (15, 16, 17)]),
            Positionable().set_points([(18, 19, 20), (21, 22, 23), (24, 25, 26)]),
        ]
    )
    assert p.get_num_points() == 0
